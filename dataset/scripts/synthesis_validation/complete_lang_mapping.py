"""Complete ISO 639-3 to Wiktionary language mapping for all 117 languages in dataset."""

COMPLETE_LANG_TO_WIKT = {
    # Germanic
    "eng": "English",
    "deu": "German",
    "nld": "Dutch",
    "swe": "Swedish",
    "nor": "Norwegian",
    "dan": "Danish",
    "isl": "Icelandic",
    "fri": "West Frisian",  # Wiktionary uses "West Frisian"
    "far": "Faroese",
    "got": "Gothic",
    "goh": "Old High German",
    "osx": "Old Saxon",
    "ang": "Old English",
    "mhd": "Middle High German",
    "gem": "Proto-Germanic",
    "non": "Old Norse",
    "nob": "Norwegian Bokmål",
    "lux": "Luxembourgish",

    # Slavic
    "rus": "Russian",
    "pol": "Polish",
    "ces": "Czech",
    "slk": "Slovak",
    "ukr": "Ukrainian",
    "bel": "Belarusian",
    "bul": "Bulgarian",
    "hbs": "Serbo-Croatian",
    "mkd": "Macedonian",
    "slv": "Slovene",
    "sor": "Lower Sorbian",  # or "Upper Sorbian" - dataset probably has both
    "chu": "Old Church Slavonic",

    # Romance
    "fra": "French",
    "ita": "Italian",
    "spa": "Spanish",
    "por": "Portuguese",
    "ron": "Romanian",
    "cat": "Catalan",
    "sar": "Sardinian",
    "nea": "Neapolitan",
    "old": "Old Occitan",
    "lat": "Latin",
    "vul": "Vulgar Latin",
    "lad": "Ladino",

    # Celtic
    "gle": "Irish",
    "cym": "Welsh",
    "bre": "Breton",
    "gla": "Scottish Gaelic",
    "sga": "Old Irish",
    "gae": "Scottish Gaelic",  # duplicate code
    "cel": "Proto-Celtic",
    "mil": "Milanese",  # Lombard dialect
    "gaw": "Gaulish",
    "gau": "Gaulish",

    # Baltic
    "lit": "Lithuanian",
    "lav": "Latvian",
    "prg": "Old Prussian",

    # Indo-Iranian
    "fas": "Persian",
    "hye": "Armenian",
    "san": "Sanskrit",
    "hin": "Hindi",
    "kas": "Kashmiri",
    "oss": "Ossetian",
    "kur": "Kurdish",
    "pal": "Middle Persian",
    "ave": "Avestan",
    "bal": "Baluchi",
    "pus": "Pashto",
    "urd": "Urdu",
    "nep": "Nepali",
    "mar": "Marathi",
    "ben": "Bengali",
    "sin": "Sinhala",  # Wiktionary uses "Sinhala"
    "pun": "Punjabi",
    "ira": "Proto-Iranian",
    "xcl": "Old Armenian",
    "hyw": "Western Armenian",
    "bho": "Bhojpuri",
    "mai": "Maithili",
    "mag": "Magahi",
    "raj": "Rajasthani",
    "kho": "Khotanese",
    "sog": "Sogdian",
    "khw": "Khowar",
    "pas": "Pashto",  # duplicate
    "par": "Parthian",

    # Greek
    "ell": "Greek",
    "grc": "Ancient Greek",
    "gmy": "Mycenaean Greek",

    # Albanian
    "sqi": "Albanian",
    "aln": "Albanian",  # Gheg Albanian - use standard Albanian

    # Anatolian
    "hit": "Hittite",
    "lyc": "Lycian",
    "xlu": "Luwian",

    # Tocharian
    "txb": "Tocharian B",
    "xto": "Tocharian A",

    # Italic (non-Latin)
    "osc": "Oscan",
    "umb": "Umbrian",

    # Other Indo-European
    "wal": "Walloon",

    # Turkic (NOT Indo-European but in dataset)
    "bak": "Bashkir",
    "tat": "Tatar",
    "kum": "Kumyk",

    # Uralic (NOT Indo-European but in dataset)
    "kal": "Kalmyk",
    "kam": "Kamba",

    # Isolates/Unknown (probably errors in dataset)
    "air": None,  # Unknown
    "ass": None,  # Unknown (maybe Assamese?)
    "aus": None,  # Unknown
    "bac": None,  # Unknown
    "dal": None,  # Unknown
    "del": None,  # Unknown (maybe Delaware?)
    "elf": None,  # Unknown
    "fle": None,  # Unknown
    "haw": None,  # Hawaiian (Austronesian)
    "lar": None,  # Unknown
    "maz": None,  # Unknown
    "meg": None,  # Unknown
    "mid": None,  # Unknown
    "wak": None,  # Unknown
    "yag": None,  # Unknown
    "zzi": None,  # Unknown (maybe Zaza?)
}

# Print statistics
if __name__ == "__main__":
    total = len(COMPLETE_LANG_TO_WIKT)
    valid = sum(1 for v in COMPLETE_LANG_TO_WIKT.values() if v is not None)
    invalid = total - valid

    print(f"Total languages: {total}")
    print(f"Valid mappings: {valid}")
    print(f"Invalid/Unknown: {invalid}")
    print()
    print("Unknown language codes:")
    for k, v in COMPLETE_LANG_TO_WIKT.items():
        if v is None:
            print(f"  {k}")
