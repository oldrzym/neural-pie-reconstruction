"""Standalone test: Wiktionary fetch with fixed regexes + template parsing."""
import asyncio, ssl, certifi, aiohttp, re

LANG_TO_WIKT = {
    'eng': 'English', 'deu': 'German', 'fra': 'French',
    'ita': 'Italian', 'spa': 'Spanish', 'nld': 'Dutch',
    'rus': 'Russian', 'swe': 'Swedish', 'pol': 'Polish',
}


def parse_template(m):
    inner = m.group(0)[2:-2]
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


async def fetch_etym(word, lang_code, session):
    lang_name = LANG_TO_WIKT.get(lang_code, '')
    if not lang_name:
        return '[No mapping]'
    url = 'https://en.wiktionary.org/w/api.php'
    params = {
        'action': 'query', 'titles': word.lower(),
        'prop': 'revisions', 'rvprop': 'content',
        'format': 'json', 'rvslots': 'main',
    }
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        if resp.status != 200:
            return f'[HTTP {resp.status}]'
        data = await resp.json()
        pages = data.get('query', {}).get('pages', {})
        for pid, page in pages.items():
            if pid == '-1':
                return '[Not found]'
            content = page.get('revisions', [{}])[0].get('slots', {}).get('main', {}).get('*', '')

            lang_pattern = rf'==\s*{re.escape(lang_name)}\s*==\n(.+?)(?=\n==[^=]|\Z)'
            lang_match = re.search(lang_pattern, content, re.DOTALL | re.IGNORECASE)
            if not lang_match:
                return f'[No {lang_name} section]'

            lang_section = lang_match.group(1)

            etym_pattern = r'===\s*Etymology[^=]*===\s*(.+?)(?=\n===|\Z)'
            etym_match = re.search(etym_pattern, lang_section, re.DOTALL)
            if etym_match:
                etym = etym_match.group(1).strip()
                etym = re.sub(r'\{\{[^}]+\}\}', parse_template, etym)
                etym = re.sub(r'\[\[([^|\]]+\|)?([^\]]+)\]\]', r'\2', etym)
                etym = re.sub(r'<[^>]+>', '', etym)
                etym = re.sub(r'\n+', ' ', etym)
                etym = re.sub(r'\s{2,}', ' ', etym)
                return etym[:400] if etym else '[Empty]'

            return '[No etymology section]'
        return '[Page error]'


async def main():
    test_words = [
        ('ride', 'eng'), ('foot', 'eng'), ('reiten', 'deu'),
        ('pied', 'fra'), ('crescere', 'ita'), ('correr', 'spa'),
        ('voet', 'nld'), ('fot', 'swe'), ('noga', 'pol'),
    ]

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    conn = aiohttp.TCPConnector(ssl=ssl_ctx)
    hdrs = {'User-Agent': 'PIE-Reconstruction/1.0 (academic research)'}

    results = []
    async with aiohttp.ClientSession(connector=conn, headers=hdrs) as session:
        for word, lang in test_words:
            await asyncio.sleep(0.5)
            result = await fetch_etym(word, lang, session)
            results.append(f'[{lang}] {word}: {result}')

    with open('_wikt_test.txt', 'w', encoding='utf-8') as f:
        for r in results:
            f.write(r + '\n\n')

asyncio.run(main())
print('Done')
