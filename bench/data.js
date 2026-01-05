window.BENCHMARK_DATA = {
  "lastUpdate": 1767595975757,
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
          "id": "98c5e223828d174e40be50fcc5422c792037f89e",
          "message": "bench: reorder and zero-pad benchmark names for dashboard sorting\n\n- Cubic benchmarks (1-3) shown before linear (4-6)\n- Each type: oneshot → construct → eval order\n- Zero-padded numbers for proper alphabetical sorting\n  - Queries: q00001, q00100, q10000\n  - Grids: g0010, g0100, g1000",
          "timestamp": "2026-01-04T22:47:51-08:00",
          "tree_id": "72abc83672667dc42f5e85f20713fc4c89a94957",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/98c5e223828d174e40be50fcc5422c792037f89e"
        },
        "date": 1767595974292,
        "tool": "julia",
        "benches": [
          {
            "name": "6_linear_eval/q10000",
            "value": 51677,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "6_linear_eval/q00001",
            "value": 44.06609485368315,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":991,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "6_linear_eval/q00100",
            "value": 561.3548387096774,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":186,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "4_linear_oneshot/q10000",
            "value": 50854,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "4_linear_oneshot/q00001",
            "value": 38.72132796780684,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":994,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "4_linear_oneshot/q00100",
            "value": 554.4308510638298,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":188,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "1_cubic_oneshot/q10000",
            "value": 92336,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "1_cubic_oneshot/q00001",
            "value": 1407.3,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "1_cubic_oneshot/q00100",
            "value": 2342.777777777778,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":9,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "3_cubic_eval/q10000",
            "value": 90723,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "3_cubic_eval/q00001",
            "value": 53.370182555780936,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":986,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "3_cubic_eval/q00100",
            "value": 972.1304347826087,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":23,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "2_cubic_construct/g1000",
            "value": 29766,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65200\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "2_cubic_construct/g0100",
            "value": 3449.125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7744\nallocs=22\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "2_cubic_construct/g0010",
            "value": 1013.0909090909091,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1664\nallocs=22\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":22,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "5_linear_construct/g1000",
            "value": 8.617617617617618,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":999,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "5_linear_construct/g0100",
            "value": 8.617617617617618,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":999,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "5_linear_construct/g0010",
            "value": 8.617617617617618,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":999,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      }
    ]
  }
}