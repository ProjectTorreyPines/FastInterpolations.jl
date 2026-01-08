window.BENCHMARK_DATA = {
  "lastUpdate": 1767897995355,
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
      }
    ]
  }
}