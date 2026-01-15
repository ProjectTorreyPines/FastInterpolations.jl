window.BENCHMARK_DATA = {
  "lastUpdate": 1768519676444,
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
      },
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
          "id": "58f53eca157c74f36503a80725549a6f3bc3c380",
          "message": "fix: update AdaptiveArrayPools dependency to correct UUID",
          "timestamp": "2026-01-05T14:04:28-08:00",
          "tree_id": "41ffa380a41e44ffba4c996ca758e82579cad53e",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/58f53eca157c74f36503a80725549a6f3bc3c380"
        },
        "date": 1767650805419,
        "tool": "julia",
        "benches": [
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1415.7,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 2302,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 86532,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 1033.9,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1664\nallocs=22\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3516.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7744\nallocs=22\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 29816,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65200\nallocs=30\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 51.605876393110435,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":987,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 952.5555555555555,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":27,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 89637,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 40.337701612903224,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":992,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 479.7755102040816,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":196,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 43982,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 12.057114228456914,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":998,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 12.057114228456914,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":998,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 12.057114228456914,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":998,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 49.36336032388664,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":988,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 491.12307692307695,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":195,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q10000",
            "value": 44043,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          }
        ]
      },
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
          "id": "5b33e3ba9188de3d3e4153d24dbb2cd94edc391e",
          "message": "fix: bump version to 0.2.1 in Project.toml",
          "timestamp": "2026-01-05T14:12:52-08:00",
          "tree_id": "01a4cadd66bb5a488c84b53a81c2ac480806f6ee",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/5b33e3ba9188de3d3e4153d24dbb2cd94edc391e"
        },
        "date": 1767651499685,
        "tool": "julia",
        "benches": [
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1397.2,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 2327.222222222222,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 91783,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 1027.7727272727273,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1664\nallocs=22\nparams={\"evals\":22,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3493.875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7744\nallocs=22\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 30563,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65200\nallocs=30\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 53.749492900608516,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":986,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 970.3333333333334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":21,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 90812,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 33.34255533199195,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":994,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 560.9281914893618,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":188,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 50896,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 9.47047047047047,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":999,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 9.47047047047047,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":999,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 9.471471471471471,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":999,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 44.21695257315842,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":991,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 568.3118279569892,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":186,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q10000",
            "value": 51711,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":10,\"time_tolerance\":0.05}"
          }
        ]
      },
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
          "id": "b79aea0308b5d78e8759c5b18b914ea7b171a9d2",
          "message": "bench: use fixed evals for consistent CI benchmark results\n\n- Skip tune!() to ensure identical measurement conditions across runs\n- Add speed-based evals categories (FAST=10K, MED=1K, SLOW=100)\n- Fast benchmarks (ns-level) get high evals to reduce timer overhead\n- Slow benchmarks (μs-level) get low evals for more samples\n- Increase samples to 100K so time limit is always the bottleneck",
          "timestamp": "2026-01-05T15:53:13-08:00",
          "tree_id": "1a6d3c214cae513aba16a36e30ee34095239bb98",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/b79aea0308b5d78e8759c5b18b914ea7b171a9d2"
        },
        "date": 1767657638651,
        "tool": "julia",
        "benches": [
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1410.956,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 2290.142,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 89559.18,
            "unit": "ns",
            "extra": "gctime=1353.72\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 1051.547,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1664\nallocs=22\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3584.051,
            "unit": "ns",
            "extra": "gctime=171.59\nmemory=7744\nallocs=22\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 29380.195,
            "unit": "ns",
            "extra": "gctime=1624.23\nmemory=65200\nallocs=30\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 52.1099,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 955.739,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 92604.04,
            "unit": "ns",
            "extra": "gctime=1350.91\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 39.1919,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 486.515,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 46010.17,
            "unit": "ns",
            "extra": "gctime=1264.955\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 20.0454,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 20.0453,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 20.0453,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 49.3869,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 497.576,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q10000",
            "value": 45874.475,
            "unit": "ns",
            "extra": "gctime=1096.1399999999999\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "48294618+mgyoo86@users.noreply.github.com",
            "name": "Min-Gu Yoo",
            "username": "mgyoo86"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f4ca62a0dfd3aac2a3502856eaf1b18b6a43e4db",
          "message": "Merge pull request #11 from ProjectTorreyPines/perf/cubic_kernel\n\n(perf) Cubic Kernel Performance Optimization",
          "timestamp": "2026-01-07T16:29:00-08:00",
          "tree_id": "2cd2a0b004e9ee5c5325c692d11c23521315dd23",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/f4ca62a0dfd3aac2a3502856eaf1b18b6a43e4db"
        },
        "date": 1767832565400,
        "tool": "julia",
        "benches": [
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1429.562,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 2079.466,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 65469.86,
            "unit": "ns",
            "extra": "gctime=1276.69\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 1060.371,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1792\nallocs=24\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3643.89,
            "unit": "ns",
            "extra": "gctime=169.727\nmemory=8528\nallocs=24\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 30036.74,
            "unit": "ns",
            "extra": "gctime=1572.94\nmemory=73224\nallocs=33\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 49.2771,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 676.8035,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 64561.055,
            "unit": "ns",
            "extra": "gctime=1245.28\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 37.467,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 452.235,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 41959.32,
            "unit": "ns",
            "extra": "gctime=1213.47\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 13.2658,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 13.2658,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 13.2658,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 45.7895,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 453.868,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q10000",
            "value": 41435.34,
            "unit": "ns",
            "extra": "gctime=1073.21\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          }
        ]
      },
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
          "id": "b4c5a9725d4806b56c61970740301cd8d05e27f7",
          "message": "fix minor bug in `ci_benchmark.jl`",
          "timestamp": "2026-01-08T10:40:36-08:00",
          "tree_id": "b42e8fdd02359666c0bae5a82235633916b07a93",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/b4c5a9725d4806b56c61970740301cd8d05e27f7"
        },
        "date": 1767897993337,
        "tool": "julia",
        "benches": [
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1412.2335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 2078.46,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 65636.24,
            "unit": "ns",
            "extra": "gctime=1497.99\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 1042.404,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1792\nallocs=24\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3662.518,
            "unit": "ns",
            "extra": "gctime=178.362\nmemory=8528\nallocs=24\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 29867.78,
            "unit": "ns",
            "extra": "gctime=1652.38\nmemory=73224\nallocs=33\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 51.6794,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 670.801,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 63851.57,
            "unit": "ns",
            "extra": "gctime=1501.7\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 37.9007,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 441.043,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 41331.66,
            "unit": "ns",
            "extra": "gctime=1271.37\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 12.7528,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 12.7527,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 12.7528,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 45.891549999999995,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 459.396,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q10000",
            "value": 41767.7,
            "unit": "ns",
            "extra": "gctime=1064.78\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/scalar_query",
            "value": 22.9789,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/vec1_query",
            "value": 51.5281,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/scalar_query",
            "value": 12.349,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/vec1_query",
            "value": 37.2124,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "48294618+mgyoo86@users.noreply.github.com",
            "name": "Min-Gu Yoo",
            "username": "mgyoo86"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "25a0415171a9406a9901be553c1cc935d6a1f0fe",
          "message": "Merge pull request #12 from ProjectTorreyPines/feat/AbstractGridSpacing\n\nfeat: Abstract Grid Spacing for O(1) uniform grid memory",
          "timestamp": "2026-01-08T11:20:42-08:00",
          "tree_id": "8ba164ce30e90ea9e41df8c33d97c04b82fb4ced",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/25a0415171a9406a9901be553c1cc935d6a1f0fe"
        },
        "date": 1767900321657,
        "tool": "julia",
        "benches": [
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1398.222,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 1890.9425,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 49304.59,
            "unit": "ns",
            "extra": "gctime=1624.74\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 996.6715,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1536\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3477.365,
            "unit": "ns",
            "extra": "gctime=218.12650000000002\nmemory=6832\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 28150.04,
            "unit": "ns",
            "extra": "gctime=1870.145\nmemory=57080\nallocs=27\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 43.1135,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 507.318,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 48095.44,
            "unit": "ns",
            "extra": "gctime=1857.56\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 36.14905,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 394.858,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 37436.23,
            "unit": "ns",
            "extra": "gctime=1444.89\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 12.7518,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 12.7518,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 12.7518,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 44.938,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 406.82050000000004,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q10000",
            "value": 37260.31,
            "unit": "ns",
            "extra": "gctime=1274.88\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/scalar_query",
            "value": 25.712,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/vec1_query",
            "value": 43.0414,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/scalar_query",
            "value": 12.0415,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/vec1_query",
            "value": 41.752,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          }
        ]
      },
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
          "id": "865ff2e5d90dcfcfe9586bdfbdb95a9972195a95",
          "message": "bump version to 0.2.2 in Project.toml",
          "timestamp": "2026-01-08T11:22:32-08:00",
          "tree_id": "cc10997a121ee529013948b88f9b1b93c74b0ae2",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/865ff2e5d90dcfcfe9586bdfbdb95a9972195a95"
        },
        "date": 1767900610138,
        "tool": "julia",
        "benches": [
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1407.8605,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 1903.6245,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 49491.97,
            "unit": "ns",
            "extra": "gctime=1771.27\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 991.566,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1536\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3489.393,
            "unit": "ns",
            "extra": "gctime=246.201\nmemory=6832\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 28922.010000000002,
            "unit": "ns",
            "extra": "gctime=2183.79\nmemory=57080\nallocs=27\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 43.3421,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 506.519,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 47893.28,
            "unit": "ns",
            "extra": "gctime=1737.15\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 36.5243,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 396.312,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 37610.619999999995,
            "unit": "ns",
            "extra": "gctime=1621.2350000000001\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 13.2659,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 13.2659,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 13.2659,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 44.366550000000004,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 412.402,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q10000",
            "value": 37925.9,
            "unit": "ns",
            "extra": "gctime=1485.38\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/scalar_query",
            "value": 25.7442,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/vec1_query",
            "value": 43.3722,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/scalar_query",
            "value": 11.733,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/vec1_query",
            "value": 39.6803,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          }
        ]
      },
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
          "id": "7b87f550da42f62430f25dabf973371fa05a07f4",
          "message": "fix: update compatibility versions for dependencies in Project.toml",
          "timestamp": "2026-01-10T23:20:18-08:00",
          "tree_id": "d2da15efbd31d213762ea20f81a0dd41f6d3663c",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/7b87f550da42f62430f25dabf973371fa05a07f4"
        },
        "date": 1768116308600,
        "tool": "julia",
        "benches": [
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1410.112,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 1891.197,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 49076.195,
            "unit": "ns",
            "extra": "gctime=1466.88\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 975.179,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1536\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3338.983,
            "unit": "ns",
            "extra": "gctime=186.221\nmemory=6832\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 27705.62,
            "unit": "ns",
            "extra": "gctime=1833.92\nmemory=57080\nallocs=27\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 42.9769,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 504.779,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 47711.765,
            "unit": "ns",
            "extra": "gctime=1472.7350000000001\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 37.8744,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 397.68,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 37586.39,
            "unit": "ns",
            "extra": "gctime=1425.85\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 12.341,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 12.341,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 12.341,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 44.5889,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 407.808,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q10000",
            "value": 37317.49,
            "unit": "ns",
            "extra": "gctime=1278.03\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/scalar_query",
            "value": 25.6487,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/vec1_query",
            "value": 42.9619,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/scalar_query",
            "value": 12.0404,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/vec1_query",
            "value": 39.8892,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          }
        ]
      },
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
          "id": "1dc39c38b01c44f30933671e4e624f411758dbd5",
          "message": "fix: update Julia compatibility version to 1.10 in Project.toml",
          "timestamp": "2026-01-10T23:27:53-08:00",
          "tree_id": "6985b76d2bd8df4b976883311e90c2a82e44be30",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/1dc39c38b01c44f30933671e4e624f411758dbd5"
        },
        "date": 1768116756378,
        "tool": "julia",
        "benches": [
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1402.11,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 1893.2535,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 49252.24,
            "unit": "ns",
            "extra": "gctime=1565.43\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 975.603,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1536\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3428.468,
            "unit": "ns",
            "extra": "gctime=213.238\nmemory=6832\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 27990.41,
            "unit": "ns",
            "extra": "gctime=1905.4099999999999\nmemory=57080\nallocs=27\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 42.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 505.995,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 47625.305,
            "unit": "ns",
            "extra": "gctime=1530.4099999999999\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 41.3963,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 397.994,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 37599.06,
            "unit": "ns",
            "extra": "gctime=1606.6\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 12.0335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 12.0335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 12.0335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 44.98755,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 409.415,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q10000",
            "value": 37425.494999999995,
            "unit": "ns",
            "extra": "gctime=1385.9450000000002\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/scalar_query",
            "value": 25.6539,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/vec1_query",
            "value": 42.78045,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/scalar_query",
            "value": 12.0405,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/vec1_query",
            "value": 38.1102,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "48294618+mgyoo86@users.noreply.github.com",
            "name": "Min-Gu Yoo",
            "username": "mgyoo86"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9df89b262bc23a816727fa7188a25f04e0075b81",
          "message": "Merge pull request #13 from ProjectTorreyPines/feat/multi_interpolant\n\nfeat: Ultra-Fast Multi-Series Interpolation via Anchored Query Optimization",
          "timestamp": "2026-01-12T10:55:08-08:00",
          "tree_id": "055b8f49655e3e98c3f7522274b3b1073225e5fb",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/9df89b262bc23a816727fa7188a25f04e0075b81"
        },
        "date": 1768244462454,
        "tool": "julia",
        "benches": [
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1405.453,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 1891.507,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 49221.375,
            "unit": "ns",
            "extra": "gctime=1636.34\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 1009.698,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1536\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3461.087,
            "unit": "ns",
            "extra": "gctime=223.375\nmemory=6832\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 28283.3,
            "unit": "ns",
            "extra": "gctime=2023.56\nmemory=57080\nallocs=27\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 43.0811,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 506.832,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 47875.11,
            "unit": "ns",
            "extra": "gctime=1660.58\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 37.8473,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 403.45,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 37473.51,
            "unit": "ns",
            "extra": "gctime=1565.2\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 12.341,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 12.341,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 12.341,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 44.7032,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 410.5985,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q10000",
            "value": 37367.08,
            "unit": "ns",
            "extra": "gctime=1390.2350000000001\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/scalar_query",
            "value": 25.7128,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/vec1_query",
            "value": 43.0471,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/scalar_query",
            "value": 12.0414,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/vec1_query",
            "value": 48.2698,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s001_q100",
            "value": 1960.546,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2352\nallocs=9\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s010_q100",
            "value": 20498.087,
            "unit": "ns",
            "extra": "gctime=972.1975\nmemory=22992\nallocs=72\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s100_q100",
            "value": 203014.13,
            "unit": "ns",
            "extra": "gctime=7197.83\nmemory=228872\nallocs=703\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s001_q100",
            "value": 1786.922,
            "unit": "ns",
            "extra": "gctime=239.51999999999998\nmemory=13096\nallocs=7\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s010_q100",
            "value": 3740.836,
            "unit": "ns",
            "extra": "gctime=485.443\nmemory=21528\nallocs=25\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s100_q100",
            "value": 21953.92,
            "unit": "ns",
            "extra": "gctime=2383.84\nmemory=105832\nallocs=205\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "48294618+mgyoo86@users.noreply.github.com",
            "name": "Min-Gu Yoo",
            "username": "mgyoo86"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6354fe96ef11cdf5fd97cdef06a3459b9ef15f82",
          "message": "Merge pull request #14 from ProjectTorreyPines/perf/cubic_multi_interpolant\n\nperf(multi-cubic): unified matrix storage with SIMD scalar kernel (10-50x speedup)",
          "timestamp": "2026-01-13T16:28:03-08:00",
          "tree_id": "6b1830d5eb3e514da425ad4ae8b2b0f5ad2ccbd4",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/6354fe96ef11cdf5fd97cdef06a3459b9ef15f82"
        },
        "date": 1768350927435,
        "tool": "julia",
        "benches": [
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1400.559,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 1891.407,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 48975.06,
            "unit": "ns",
            "extra": "gctime=1437.99\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 984.641,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1536\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3383.517,
            "unit": "ns",
            "extra": "gctime=186.529\nmemory=6832\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 27856.79,
            "unit": "ns",
            "extra": "gctime=1860.68\nmemory=57080\nallocs=27\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 42.7229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 505.045,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 47489.095,
            "unit": "ns",
            "extra": "gctime=1440.6\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 36.0253,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 393.375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 37271.66,
            "unit": "ns",
            "extra": "gctime=1400.02\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 12.7519,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 12.7519,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 12.7519,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 44.7076,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 405.067,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q10000",
            "value": 37179.59,
            "unit": "ns",
            "extra": "gctime=1256.75\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/scalar_query",
            "value": 25.655,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/vec1_query",
            "value": 42.6938,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/scalar_query",
            "value": 11.7329,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/vec1_query",
            "value": 38.2556,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s001_q100",
            "value": 2146.6220000000003,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2080\nallocs=5\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s010_q100",
            "value": 14802.025,
            "unit": "ns",
            "extra": "gctime=372.369\nmemory=16368\nallocs=7\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s100_q100",
            "value": 138028.65,
            "unit": "ns",
            "extra": "gctime=3010.73\nmemory=160368\nallocs=7\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s001_q100",
            "value": 2016.713,
            "unit": "ns",
            "extra": "gctime=0\nmemory=992\nallocs=4\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s001_q100_scalar_loop",
            "value": 3732.795,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s010_q100",
            "value": 3623.381,
            "unit": "ns",
            "extra": "gctime=182.522\nmemory=9424\nallocs=22\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s010_q100_scalar_loop",
            "value": 4870.1355,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s100_q100",
            "value": 19435.85,
            "unit": "ns",
            "extra": "gctime=1951.76\nmemory=93728\nallocs=202\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s100_q100_scalar_loop",
            "value": 5554.89,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "48294618+mgyoo86@users.noreply.github.com",
            "name": "Min-Gu Yoo",
            "username": "mgyoo86"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "43642294456b3caee873de9dd3f3adc078d8dd68",
          "message": "Merge pull request #15 from ProjectTorreyPines/refac/file_structure\n\nRefactor: Modular File Structure with Consistent Organization",
          "timestamp": "2026-01-13T22:38:30-08:00",
          "tree_id": "7b945041c74606ddbb9843356a813e2ad8185b46",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/43642294456b3caee873de9dd3f3adc078d8dd68"
        },
        "date": 1768373083136,
        "tool": "julia",
        "benches": [
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1400.297,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 1879.171,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 49053.13,
            "unit": "ns",
            "extra": "gctime=1457.02\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 1024.955,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1536\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3486.178,
            "unit": "ns",
            "extra": "gctime=233.927\nmemory=6832\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 27611.48,
            "unit": "ns",
            "extra": "gctime=1880.61\nmemory=57080\nallocs=27\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 51.9721,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 512.508,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 47553.03,
            "unit": "ns",
            "extra": "gctime=1480.16\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 37.5441,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 396.861,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 37334.74,
            "unit": "ns",
            "extra": "gctime=1432.97\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 12.0335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 12.0335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 12.0335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 44.1205,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 407.887,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q10000",
            "value": 37273.005000000005,
            "unit": "ns",
            "extra": "gctime=1298.03\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/scalar_query",
            "value": 25.6519,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/vec1_query",
            "value": 51.8439,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/scalar_query",
            "value": 11.7329,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/vec1_query",
            "value": 38.92415,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s001_q100",
            "value": 2185.693,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2080\nallocs=5\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s010_q100",
            "value": 15695.718,
            "unit": "ns",
            "extra": "gctime=612.785\nmemory=16368\nallocs=7\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s100_q100",
            "value": 141069.635,
            "unit": "ns",
            "extra": "gctime=3975.5\nmemory=160368\nallocs=7\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s001_q100",
            "value": 2021.321,
            "unit": "ns",
            "extra": "gctime=0\nmemory=992\nallocs=4\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s001_q100_scalar_loop",
            "value": 3746.888,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s010_q100",
            "value": 3772.626,
            "unit": "ns",
            "extra": "gctime=218.598\nmemory=9424\nallocs=22\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s010_q100_scalar_loop",
            "value": 4902.0545,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s100_q100",
            "value": 19597.464999999997,
            "unit": "ns",
            "extra": "gctime=1986.11\nmemory=93728\nallocs=202\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s100_q100_scalar_loop",
            "value": 5505.48,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "48294618+mgyoo86@users.noreply.github.com",
            "name": "Min-Gu Yoo",
            "username": "mgyoo86"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f87c9c6b9d91a71b777982e6948ee9799f17e45c",
          "message": "Merge pull request #16 from ProjectTorreyPines/refac/series_interpolant\n\n(refac): Complete SeriesInterpolant Refactoring",
          "timestamp": "2026-01-14T15:13:10-08:00",
          "tree_id": "64d97dae5b522934cd9d03423e563eea52db90b3",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/f87c9c6b9d91a71b777982e6948ee9799f17e45c"
        },
        "date": 1768432841719,
        "tool": "julia",
        "benches": [
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1403.9325,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 1895.3805,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 49646.259999999995,
            "unit": "ns",
            "extra": "gctime=1961\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 992.1595,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1536\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3406.168,
            "unit": "ns",
            "extra": "gctime=190.686\nmemory=6832\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 27823.275,
            "unit": "ns",
            "extra": "gctime=1924.59\nmemory=57080\nallocs=27\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 42.8698,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 504.811,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 47552.07,
            "unit": "ns",
            "extra": "gctime=1463.4850000000001\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 38.0098,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 397.541,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 37634.35,
            "unit": "ns",
            "extra": "gctime=1764.89\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 12.341,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 12.341,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 12.341,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 44.1953,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 409.734,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q10000",
            "value": 37486.345,
            "unit": "ns",
            "extra": "gctime=1471.6399999999999\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/scalar_query",
            "value": 25.6538,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/vec1_query",
            "value": 43.2527,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/scalar_query",
            "value": 12.0424,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/vec1_query",
            "value": 40.0596,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s001_q100",
            "value": 2161.3965,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2064\nallocs=6\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s010_q100",
            "value": 14981.3055,
            "unit": "ns",
            "extra": "gctime=400.42100000000005\nmemory=16352\nallocs=8\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s100_q100",
            "value": 138568.49,
            "unit": "ns",
            "extra": "gctime=3123.72\nmemory=160352\nallocs=8\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s001_q100",
            "value": 2013.714,
            "unit": "ns",
            "extra": "gctime=0\nmemory=992\nallocs=4\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s001_q100_scalar_loop",
            "value": 2980.6795,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s010_q100",
            "value": 3637.2335000000003,
            "unit": "ns",
            "extra": "gctime=184.599\nmemory=9424\nallocs=22\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s010_q100_scalar_loop",
            "value": 4511.189,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s100_q100",
            "value": 19817.34,
            "unit": "ns",
            "extra": "gctime=1942.72\nmemory=93728\nallocs=202\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s100_q100_scalar_loop",
            "value": 5356.335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          }
        ]
      },
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
          "id": "66a880ca41b8e33101cdc94541619bde0a36bfb4",
          "message": "docs: remove installation instructions from README",
          "timestamp": "2026-01-14T15:14:46-08:00",
          "tree_id": "e8afaacc22e4920d9a05562c495a39683faddb89",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/66a880ca41b8e33101cdc94541619bde0a36bfb4"
        },
        "date": 1768433220187,
        "tool": "julia",
        "benches": [
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1407.229,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 1886.353,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 49130.62,
            "unit": "ns",
            "extra": "gctime=1473.86\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 988.198,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1536\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3361.084,
            "unit": "ns",
            "extra": "gctime=187.601\nmemory=6832\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 27349.79,
            "unit": "ns",
            "extra": "gctime=1780.18\nmemory=57080\nallocs=27\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 42.79,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 506.267,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 47582.29,
            "unit": "ns",
            "extra": "gctime=1472.35\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 37.398,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 397.804,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 37669.88,
            "unit": "ns",
            "extra": "gctime=1425.7649999999999\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 12.3411,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 12.3411,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 12.3411,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 44.5053,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 407.277,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q10000",
            "value": 37272.965,
            "unit": "ns",
            "extra": "gctime=1279.49\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/scalar_query",
            "value": 25.7111,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/vec1_query",
            "value": 42.7571,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/scalar_query",
            "value": 12.0415,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/vec1_query",
            "value": 37.4721,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s001_q100",
            "value": 2178.1014999999998,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2064\nallocs=6\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s010_q100",
            "value": 14932.176,
            "unit": "ns",
            "extra": "gctime=388.065\nmemory=16352\nallocs=8\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s100_q100",
            "value": 138726.73,
            "unit": "ns",
            "extra": "gctime=3164.42\nmemory=160352\nallocs=8\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s001_q100",
            "value": 2036.5120000000002,
            "unit": "ns",
            "extra": "gctime=0\nmemory=992\nallocs=4\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s001_q100_scalar_loop",
            "value": 2944.208,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s010_q100",
            "value": 3658.0175,
            "unit": "ns",
            "extra": "gctime=184.01850000000002\nmemory=9424\nallocs=22\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s010_q100_scalar_loop",
            "value": 4528.272999999999,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s100_q100",
            "value": 19690.955,
            "unit": "ns",
            "extra": "gctime=1963.07\nmemory=93728\nallocs=202\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s100_q100_scalar_loop",
            "value": 5356.82,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "48294618+mgyoo86@users.noreply.github.com",
            "name": "Min-Gu Yoo",
            "username": "mgyoo86"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e1c0f8664cff3d461adde04b8918f90ce9cb1a4a",
          "message": "Merge pull request #17 from ProjectTorreyPines/feat/Deriv3\n\n(feat): Third Derivative (Deriv3) Support",
          "timestamp": "2026-01-15T13:49:14-08:00",
          "tree_id": "8dd91f1078713eb86a76548dc2a32baef2ad58da",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/e1c0f8664cff3d461adde04b8918f90ce9cb1a4a"
        },
        "date": 1768514134540,
        "tool": "julia",
        "benches": [
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1405.0955,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 1897.187,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 48980.165,
            "unit": "ns",
            "extra": "gctime=1422.045\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 1028.348,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1536\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3392.876,
            "unit": "ns",
            "extra": "gctime=187.96\nmemory=6832\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 27408.7,
            "unit": "ns",
            "extra": "gctime=1743.36\nmemory=57080\nallocs=27\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 42.8499,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 504.952,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 47603.84,
            "unit": "ns",
            "extra": "gctime=1404.52\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 36.7435,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 395.447,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 37333.9,
            "unit": "ns",
            "extra": "gctime=1399.91\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 13.3168,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 13.3168,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 13.3168,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 44.0801,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 417.86800000000005,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q10000",
            "value": 38078.744999999995,
            "unit": "ns",
            "extra": "gctime=1230.09\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/scalar_query",
            "value": 25.7139,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/vec1_query",
            "value": 42.9,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/scalar_query",
            "value": 12.0414,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/vec1_query",
            "value": 41.30855,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s001_q100",
            "value": 2164.756,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2064\nallocs=6\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s010_q100",
            "value": 14910.947,
            "unit": "ns",
            "extra": "gctime=393.664\nmemory=16352\nallocs=8\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s100_q100",
            "value": 138714.67,
            "unit": "ns",
            "extra": "gctime=3173.11\nmemory=160352\nallocs=8\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s001_q100",
            "value": 2099.8145,
            "unit": "ns",
            "extra": "gctime=0\nmemory=992\nallocs=4\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s001_q100_scalar_loop",
            "value": 3123.376,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s010_q100",
            "value": 3893.132,
            "unit": "ns",
            "extra": "gctime=186.553\nmemory=9424\nallocs=22\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s010_q100_scalar_loop",
            "value": 4495.282,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s100_q100",
            "value": 19874.75,
            "unit": "ns",
            "extra": "gctime=1968.23\nmemory=93728\nallocs=202\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s100_q100_scalar_loop",
            "value": 5315.61,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "48294618+mgyoo86@users.noreply.github.com",
            "name": "Min-Gu Yoo",
            "username": "mgyoo86"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8ecae10fff002dbe4659fec26b9329e4c5ab2b7c",
          "message": "Merge pull request #18 from ProjectTorreyPines/feat/bc_array\n\n(feat): Per-Series Boundary Conditions for SeriesInterpolant",
          "timestamp": "2026-01-15T15:21:31-08:00",
          "tree_id": "f0b56d685fa4cf6e196cd4d6494cc81395f37ad5",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/8ecae10fff002dbe4659fec26b9329e4c5ab2b7c"
        },
        "date": 1768519674350,
        "tool": "julia",
        "benches": [
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1407.545,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 1896.921,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 49208.43,
            "unit": "ns",
            "extra": "gctime=1579.3600000000001\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 1003.318,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1536\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3436.526,
            "unit": "ns",
            "extra": "gctime=199.809\nmemory=6832\nallocs=20\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 28506.64,
            "unit": "ns",
            "extra": "gctime=1820.8600000000001\nmemory=57080\nallocs=27\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 43.9633,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 510.785,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 47895.37,
            "unit": "ns",
            "extra": "gctime=1597.89\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 38.5611,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 398.836,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 37762.229999999996,
            "unit": "ns",
            "extra": "gctime=1780.835\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 12.0335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 12.0336,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 12.0335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 45.0304,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 417.099,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "6_linear_eval/q10000",
            "value": 38152.36,
            "unit": "ns",
            "extra": "gctime=1332.89\nmemory=80072\nallocs=3\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/scalar_query",
            "value": 25.7111,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_range/vec1_query",
            "value": 43.9413,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/scalar_query",
            "value": 12.0415,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "7_cubic_vec/vec1_query",
            "value": 39.4278,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"evals\":10000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s001_q100",
            "value": 2166.119,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2064\nallocs=6\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s010_q100",
            "value": 15063.939999999999,
            "unit": "ns",
            "extra": "gctime=416.509\nmemory=16352\nallocs=8\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/construct_s100_q100",
            "value": 139433.48,
            "unit": "ns",
            "extra": "gctime=3334.55\nmemory=160352\nallocs=8\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s001_q100",
            "value": 2063.722,
            "unit": "ns",
            "extra": "gctime=0\nmemory=992\nallocs=4\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s001_q100_scalar_loop",
            "value": 3117.982,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s010_q100",
            "value": 3755.012,
            "unit": "ns",
            "extra": "gctime=191.999\nmemory=9424\nallocs=22\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s010_q100_scalar_loop",
            "value": 4526.35,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":1000,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s100_q100",
            "value": 19717.675,
            "unit": "ns",
            "extra": "gctime=1944.2350000000001\nmemory=93728\nallocs=202\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          },
          {
            "name": "8_cubic_multi/eval_s100_q100_scalar_loop",
            "value": 5318.35,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"evals\":100,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":100000,\"seconds\":10,\"time_tolerance\":0.05}"
          }
        ]
      }
    ]
  }
}