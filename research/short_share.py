"""v2 momentum short share (live 0.4) on the live 4x book; the team sleeve is left as vendored."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from research.harness import Lab  # noqa: E402
from trend_guard.strategy.book import book_components  # noqa: E402

lab = Lab()
books = {"book (0.4)": lab.book}
for ss in [float(x) for x in (sys.argv[1:] or ["0.2", "0.3", "0.5"])]:
    cfg = lab.cfg.book.model_copy(
        update={"momentum": lab.cfg.book.momentum.model_copy(update={"short_share": ss})}
    )
    books[f"short_share {ss}"] = sum(book_components(lab.close, lab.qv, cfg, True).values())
lab.report(books)
