"""Standalone Wiktionary etymology fetcher with retry logic for 429 errors."""
import asyncio
import aiohttp
import ssl
import certifi
import json
import re
import time
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent.parent.parent / "splits" / "iecor_kaikki_koebler_normalized"
ETYMOLOGY_DICT = BASE_DIR / "wiktionary_etymologies.json"

# Rate limiting with aggressive backoff
INITIAL_DELAY = 0.5           # Start with 0.5 second delay (was 1.0)
MAX_DELAY = 30.0              # Max 30 seconds between requests
CONCURRENT_REQUESTS = 2       # 2 parallel requests (was 1) - safer than 3+
BACKOFF_MULTIPLIER = 2.0      # Double delay on each 429
MAX_RETRIES = 5               # Retry up to 5 times per word

# Complete language mapping (ISO 639-3 to Wiktionary language names)
# Import from complete_lang_mapping.py
from complete_lang_mapping import COMPLETE_LANG_TO_WIKT as LANG_TO_WIKT


class WiktionaryFetcher:
    def __init__(self):
        self.cache = self.load_cache()
        self.current_delay = INITIAL_DELAY
        self.session = None
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.headers = {'User-Agent': 'PIE-Reconstruction/1.0 (academic research)'}

    def load_cache(self):
        """Load existing etymology cache."""
        if ETYMOLOGY_DICT.exists():
            with open(ETYMOLOGY_DICT, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_cache(self):
        """Save etymology cache to disk."""
        with open(ETYMOLOGY_DICT, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def parse_template(self, match):
        """Parse Wiktionary template {{...}}."""
        inner = match.group(0)[2:-2]
        parts = inner.split('|')
        word = ''
        trans = ''
        for p in parts[1:]:
            if p.startswith('t=') or p.startswith('gloss='):
                trans = p.split('=', 1)[1]
            elif '=' not in p:
                word = p
        if trans:
            return f'{word} ("{trans}")'
        return word

    async def fetch_etymology(self, word, lang_code, retry_count=0):
        """
        Fetch etymology for a word from Wiktionary with retry logic.

        Returns:
            str: Etymology text or error message
        """
        key = f"{lang_code}|{word}"

        # Check cache first
        if key in self.cache:
            return self.cache[key]

        # Check if language is supported
        lang_name = LANG_TO_WIKT.get(lang_code)
        if not lang_name:
            result = f"[No Wiktionary mapping for {lang_code}]"
            self.cache[key] = result
            return result

        # Apply current delay
        await asyncio.sleep(self.current_delay)

        try:
            url = 'https://en.wiktionary.org/w/api.php'
            params = {
                'action': 'query',
                'titles': word.lower(),
                'prop': 'revisions',
                'rvprop': 'content',
                'format': 'json',
                'rvslots': 'main',
            }

            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                # Handle 429 rate limit
                if resp.status == 429:
                    if retry_count < MAX_RETRIES:
                        # Exponential backoff
                        wait_time = self.current_delay * (BACKOFF_MULTIPLIER ** retry_count)
                        wait_time = min(wait_time, MAX_DELAY)

                        print(f"\n⚠️  HTTP 429 for {key}, retrying in {wait_time:.1f}s (attempt {retry_count + 1}/{MAX_RETRIES})")

                        # Increase global delay
                        self.current_delay = min(self.current_delay * 1.5, MAX_DELAY)

                        await asyncio.sleep(wait_time)
                        return await self.fetch_etymology(word, lang_code, retry_count + 1)
                    else:
                        result = "[Wiktionary HTTP 429 - max retries exceeded]"
                        self.cache[key] = result
                        return result

                if resp.status != 200:
                    result = f"[HTTP {resp.status}]"
                    self.cache[key] = result
                    return result

                data = await resp.json()
                pages = data.get('query', {}).get('pages', {})

                for page_id, page in pages.items():
                    if page_id == '-1':
                        result = "[Word not found in Wiktionary]"
                        self.cache[key] = result
                        return result

                    content = page.get('revisions', [{}])[0].get('slots', {}).get('main', {}).get('*', '')

                    # Find language section
                    lang_pattern = rf"==\s*{re.escape(lang_name)}\s*==\n(.+?)(?=\n==[^=]|\Z)"
                    lang_match = re.search(lang_pattern, content, re.DOTALL | re.IGNORECASE)

                    if not lang_match:
                        result = f"[No {lang_name} section found]"
                        self.cache[key] = result
                        return result

                    lang_section = lang_match.group(1)

                    # Find etymology section
                    etym_pattern = r"===\s*Etymology[^=]*===\s*(.+?)(?=\n===|\Z)"
                    etym_match = re.search(etym_pattern, lang_section, re.DOTALL)

                    if not etym_match:
                        result = "[No etymology section found]"
                        self.cache[key] = result
                        return result

                    etymology = etym_match.group(1).strip()

                    # Clean up etymology
                    etymology = re.sub(r'\{\{[^}]+\}\}', self.parse_template, etymology)
                    etymology = re.sub(r'\[\[([^|\]]+\|)?([^\]]+)\]\]', r'\2', etymology)
                    etymology = re.sub(r'<[^>]+>', '', etymology)
                    etymology = re.sub(r'\n+', ' ', etymology)
                    etymology = re.sub(r'\s{2,}', ' ', etymology)
                    etymology = etymology[:500]

                    result = etymology if etymology else "[Empty etymology]"
                    self.cache[key] = result

                    # Decrease delay on success
                    self.current_delay = max(INITIAL_DELAY, self.current_delay * 0.95)

                    return result

                result = "[Page structure error]"
                self.cache[key] = result
                return result

        except asyncio.TimeoutError:
            result = "[Wiktionary timeout]"
            self.cache[key] = result
            return result
        except Exception as e:
            result = f"[Error: {str(e)}]"
            self.cache[key] = result
            return result

    async def fetch_batch(self, word_lang_pairs):
        """
        Fetch etymologies for a batch of (word, lang) pairs.

        Args:
            word_lang_pairs: List of (word, lang_code) tuples

        Returns:
            dict: {lang|word: etymology}
        """
        connector = aiohttp.TCPConnector(ssl=self.ssl_context)
        async with aiohttp.ClientSession(connector=connector, headers=self.headers) as session:
            self.session = session

            # Semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

            async def fetch_with_semaphore(word, lang, pbar):
                async with semaphore:
                    key = f"{lang}|{word}"
                    etym = await self.fetch_etymology(word, lang)
                    pbar.update(1)
                    return key, etym

            results = {}
            pbar = tqdm(total=len(word_lang_pairs), desc="Fetching etymologies")

            # Process in chunks to save periodically
            chunk_size = 50
            for i in range(0, len(word_lang_pairs), chunk_size):
                chunk = word_lang_pairs[i:i + chunk_size]

                # Fetch chunk in parallel (limited by semaphore)
                tasks = [fetch_with_semaphore(word, lang, pbar) for word, lang in chunk]
                chunk_results = await asyncio.gather(*tasks)

                for key, etym in chunk_results:
                    results[key] = etym
                    self.cache[key] = etym

                # Save cache after each chunk
                self.save_cache()

            pbar.close()
            return results


async def main():
    """Main entry point for standalone usage."""
    fetcher = WiktionaryFetcher()

    # Example usage
    test_pairs = [
        ('ride', 'eng'),
        ('foot', 'eng'),
        ('reiten', 'deu'),
        ('pied', 'fra'),
        ('noga', 'pol'),
        ('zima', 'rus'),
        ('krossa', 'swe'),
    ]

    print(f"Testing Wiktionary fetcher with {len(test_pairs)} words...")
    print(f"Initial delay: {INITIAL_DELAY}s")
    print(f"Max delay: {MAX_DELAY}s")
    print(f"Max retries: {MAX_RETRIES}\n")

    results = await fetcher.fetch_batch(test_pairs)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")

    for key, etym in results.items():
        print(f"{key}:")
        print(f"  {etym[:200]}...")
        print()

    print(f"\n✓ Cache saved to: {ETYMOLOGY_DICT}")


if __name__ == "__main__":
    asyncio.run(main())
