window.BENCHMARK_DATA = {
  "lastUpdate": 1767595139468,
  "repoUrl": "https://github.com/ProjectTorreyPines/FastInterpolations.jl",
  "entries": {
    "FastInterpolations.jl Benchmarks": [
      {
        "commit": {
          "author": {
            "name": "Min-Gu Yoo",
            "username": "mgyoo86",
            "email": "mgyoo86@gmail.com"
          },
          "committer": {
            "name": "Min-Gu Yoo",
            "username": "mgyoo86",
            "email": "mgyoo86@gmail.com"
          },
          "id": "d2ac3f72f5cce6da3a5791906ecf2fadda6f6e39",
          "message": "refactor: simplify CI benchmarks to 18 core tests\n\n- Remove package comparison benchmarks (Interpolations.jl, DataInterpolations.jl)\n- Focus on FastInterpolations regression detection only\n- Benchmark structure:\n  - oneshot: linear/cubic × q1/q100/q10000 (6)\n  - construct: linear/cubic × g10/g100/g1000 (6)\n  - eval: linear/cubic × q1/q100/q10000 (6)\n- Add benchmark badge to README",
          "timestamp": "2026-01-05T06:28:57Z",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/d2ac3f72f5cce6da3a5791906ecf2fadda6f6e39"
        },
        "date": 1767595138235,
        "tool": "julia",
        "benches": [
          {
            "name": "construct/cubic_g10",
            "value": 1017.2,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1664\nallocs=22\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":15,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "construct/linear_g1000",
            "value": 13.327327327327327,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":999,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "construct/linear_g10",
            "value": 13.318318318318319,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":999,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "construct/linear_g100",
            "value": 13.318318318318319,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":999,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "construct/cubic_g100",
            "value": 3408.875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7744\nallocs=22\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "construct/cubic_g1000",
            "value": 28654,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65200\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/cubic_q10000",
            "value": 86712,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/linear_q10000",
            "value": 43982,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/cubic_q100",
            "value": 2301,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":9,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/cubic_q1",
            "value": 1415.7,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/linear_q1",
            "value": 39.04389505549949,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":991,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/linear_q100",
            "value": 482.6461538461538,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":195,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/cubic_q10000",
            "value": 89798,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/linear_q10000",
            "value": 44023,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/cubic_q100",
            "value": 963.6363636363636,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":22,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/cubic_q1",
            "value": 51.750507099391484,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":986,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/linear_q1",
            "value": 49.31275303643725,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":988,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/linear_q100",
            "value": 500.680412371134,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":194,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      }
    ]
  }
}