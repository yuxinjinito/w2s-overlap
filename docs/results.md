# Results

Research in progress. Numbers are not published here yet.

What is fixed already is how a result gets reported. Each run writes a `summary.json`
with its own base and gold-label anchors, and any number that lands here will be a mean
over three training seeds at a fixed data seed, quoted against those anchors. Offline
screen metrics gate which variants earn a run and will not appear here as results.

See [`method.md`](method.md) for what is being measured and
[`reproducing.md`](reproducing.md) for how to run it.