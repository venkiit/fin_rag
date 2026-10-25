import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from index import LocalEmbeddings, build_or_load_index

store = build_or_load_index(sys.argv[1], LocalEmbeddings())
docs = list(store.docstore._dict.values())          # every chunk stored in the index
for term in sys.argv[2:]:
    pages = sorted({d.metadata["page"] for d in docs if term in d.page_content})
    print(f"{term!r:>12} -> pages {pages}")