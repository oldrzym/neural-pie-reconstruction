# Data licenses and provenance

The root Apache-2.0 license applies to original software only. It does not
replace the licenses of source datasets or external projects.

## IE-CoR

- Source: https://github.com/lexibank/iecor
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Required attribution: cite the IE-CoR release and the associated publication.

The processed `dataset/processed/iecor.csv` file and IE-CoR rows in public
combined splits retain this source license and attribution requirement.

## Kaikki and Wiktionary

- Source: https://kaikki.org/dictionary/Proto-Indo-European/
- Upstream content: English Wiktionary
- License: Creative Commons Attribution-ShareAlike 4.0 International
- License information: https://en.wiktionary.org/wiki/Wiktionary:Copyrights

Kaikki-derived rows and validated synthetic rows that use Wiktionary evidence
are distributed under CC BY-SA 4.0. They must retain attribution, license
notices, and an indication that the material was transformed.

The exact Kaikki dump date was not preserved in the research files. This is a
known provenance limitation of the release.

## EtymologyDB

- Source: https://github.com/droher/etymology-db
- Data license declared upstream: Creative Commons ShareAlike 3.0
- Code license declared upstream: Apache-2.0

EtymologyDB-derived rows retain the upstream data terms. The public manifests
label these rows with `source=etymology_db`; blank source values in the internal
research artifact were repaired during export.

## Public combined variants

The selection, release identifiers, manifests, and original additions to the
public combined variants are made available under CC BY-SA 4.0. Individual
source rows continue to retain their original licenses and attribution terms.

## Material not distributed

- Koebler-derived rows: no open redistribution license was identified.
- Starling-derived rows and cached HTML: no verified redistribution license.
- Raw OpenAI batch request/response logs: excluded because they are operational
  artifacts and may contain provider identifiers and copied source snippets.
- Model checkpoints and pickle vocabularies: excluded from the code repository.

The Koebler and Starling parsers are original software. Their presence does not
grant rights to redistribute content obtained from those websites.
