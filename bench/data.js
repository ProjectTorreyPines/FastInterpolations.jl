window.BENCHMARK_DATA = {
  "lastUpdate": 1767598620896,
  "repoUrl": "https://github.com/ProjectTorreyPines/FastInterpolations.jl",
  "entries": {
    "FastInterpolations.jl Benchmarks": [
      {
        "commit": {
          "author": {
            "email": "mgyoo86@gmail.com",
            "name": "Min-Gu Yoo",
            "username": "mgyoo86"
          },
          "committer": {
            "email": "mgyoo86@gmail.com",
            "name": "Min-Gu Yoo",
            "username": "mgyoo86"
          },
          "distinct": true,
          "id": "74917fd5be50b195074ef29bdaa6c5191cf734f9",
          "message": "fix: use BenchmarkTools.save with sorted JSON keys for dashboard ordering\n\n- Use BenchmarkTools.save() to preserve complex nested structure\n- Add sort_keys_recursive() to sort all Dict keys alphabetically\n- Use OrderedDict to maintain key order in JSON output\n- Ensures github-action-benchmark compatibility while fixing display order",
          "timestamp": "2026-01-04T23:31:40-08:00",
          "tree_id": "313c465199406703c592de2ee1c45999c7f5c548",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/74917fd5be50b195074ef29bdaa6c5191cf734f9"
        },
        "date": 1767598619649,
        "tool": "julia",
        "benches": [
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1427.7,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 2311,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 86822,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 1037.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1664\nallocs=22\nparams={\"evals\":12,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3458.875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7744\nallocs=22\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 29715,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65200\nallocs=30\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 51.93110435663627,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":987,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 954.3703703703703,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":27,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 89998,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 40.36693548387097,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":992,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 480.3316326530612,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":196,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 44012,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 12.365365365365365,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":999,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 12.365365365365365,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":999,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 12.365365365365365,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":999,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 48.94736842105263,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":988,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 502.7846153846154,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":195,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q10000",
            "value": 44073,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          }
        ]
      }
    ]
  }
}