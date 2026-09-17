"""Test script to check Wiktionary language header names."""
import requests
import re

def check_word(word):
    """Fetch a word from Wiktionary and show all language headers."""
    url = 'https://en.wiktionary.org/w/api.php'
    params = {
        'action': 'query',
        'titles': word,
        'prop': 'revisions',
        'rvprop': 'content',
        'format': 'json',
        'rvslots': 'main',
    }

    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    pages = data['query']['pages']

    for page_id, page in pages.items():
        if page_id == '-1':
            print(f"Word '{word}' not found")
            return

        content = page['revisions'][0]['slots']['main']['*']

        # Find all level-2 headers (==Language==)
        headers = re.findall(r'\n==(.*?)==\n', content)

        print(f"\nWord: {word}")
        print(f"Language sections found:")
        for h in headers:
            print(f"  =={h}==")

        return headers


# Test words from different languages
test_words = [
    ('bitter', 'Should have Old English'),
    ('foot', 'Should have English'),
    ('Fuß', 'Should have German'),
    ('ride', 'Should have English + maybe Old Norse'),
    ('beran', 'Should have Old English'),
]

print("="*60)
print("WIKTIONARY LANGUAGE HEADER TEST")
print("="*60)

for word, description in test_words:
    print(f"\n{description}")
    check_word(word.lower())
