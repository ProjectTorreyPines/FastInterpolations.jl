window.BENCHMARK_DATA = {
  "lastUpdate": 1767594752516,
  "repoUrl": "https://github.com/ProjectTorreyPines/FastInterpolations.jl",
  "entries": {
    "FastInterpolations.jl Benchmarks": [
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
          "id": "40763a048d345674970d56e8137c61957f9d0650",
          "message": "Merge pull request #10 from ProjectTorreyPines/feat/benchmark\n\nfeat: add CI benchmark workflow with historical tracking",
          "timestamp": "2026-01-04T19:18:39-08:00",
          "tree_id": "65e46078106147e8d9f53997331d2c87ce3fed52",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/40763a048d345674970d56e8137c61957f9d0650"
        },
        "date": 1767583572049,
        "tool": "julia",
        "benches": [
          {
            "name": "construct/cubic",
            "value": 3471.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7744\nallocs=22\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "construct/linear",
            "value": 12.364364364364365,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":999,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/Interpolations_1000",
            "value": 22813,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23640\nallocs=79\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/DataInterp_100",
            "value": 12132,
            "unit": "ns",
            "extra": "gctime=0\nmemory=17376\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/Interpolations_100",
            "value": 9224,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16496\nallocs=78\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":3,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/DataInterp_10000",
            "value": 747090,
            "unit": "ns",
            "extra": "gctime=0\nmemory=96520\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/FastInterp_10000",
            "value": 86280,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/Interpolations_10000",
            "value": 155700,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95640\nallocs=79\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/Interpolations_100000",
            "value": 1474403,
            "unit": "ns",
            "extra": "gctime=0\nmemory=815640\nallocs=79\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/Interpolations_10",
            "value": 7825.875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15712\nallocs=78\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":4,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/FastInterp_100000",
            "value": 859349,
            "unit": "ns",
            "extra": "gctime=0\nmemory=800072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/FastInterp_10",
            "value": 1507.7,
            "unit": "ns",
            "extra": "gctime=0\nmemory=144\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/DataInterp_1000",
            "value": 78446,
            "unit": "ns",
            "extra": "gctime=0\nmemory=24520\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/DataInterp_100000",
            "value": 7413290,
            "unit": "ns",
            "extra": "gctime=0\nmemory=816520\nallocs=51\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/FastInterp_100",
            "value": 2309.8888888888887,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":9,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/DataInterp_10",
            "value": 5271.416666666666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16592\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":6,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/FastInterp_1",
            "value": 1419.7,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/FastInterp_1000",
            "value": 10138,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/Interpolations_1",
            "value": 7674.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15632\nallocs=78\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":4,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "compare/DataInterp_1",
            "value": 4654.428571428572,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16512\nallocs=50\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/linear_10",
            "value": 76.96086508753862,
            "unit": "ns",
            "extra": "gctime=0\nmemory=144\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":971,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/linear_1000",
            "value": 4481.142857142857,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/cubic_100",
            "value": 2312.1111111111113,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":9,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/cubic_10",
            "value": 1507.7,
            "unit": "ns",
            "extra": "gctime=0\nmemory=144\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/cubic_1000",
            "value": 10137,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/linear_100",
            "value": 480.6887755102041,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":196,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/linear_10",
            "value": 90.27244258872652,
            "unit": "ns",
            "extra": "gctime=0\nmemory=144\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":958,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/linear_1000",
            "value": 4482.571428571428,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":7,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/cubic_100",
            "value": 960.4090909090909,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":22,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/cubic_10",
            "value": 132.4326160815402,
            "unit": "ns",
            "extra": "gctime=0\nmemory=144\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":883,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/cubic_1000",
            "value": 9198,
            "unit": "ns",
            "extra": "gctime=0\nmemory=8072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/linear_100",
            "value": 504.26153846153846,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":195,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
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
          "id": "d2ac3f72f5cce6da3a5791906ecf2fadda6f6e39",
          "message": "refactor: simplify CI benchmarks to 18 core tests\n\n- Remove package comparison benchmarks (Interpolations.jl, DataInterpolations.jl)\n- Focus on FastInterpolations regression detection only\n- Benchmark structure:\n  - oneshot: linear/cubic × q1/q100/q10000 (6)\n  - construct: linear/cubic × g10/g100/g1000 (6)\n  - eval: linear/cubic × q1/q100/q10000 (6)\n- Add benchmark badge to README",
          "timestamp": "2026-01-04T22:30:23-08:00",
          "tree_id": "6cdd367d0e0859576cda9df726efa372667cdb27",
          "url": "https://github.com/ProjectTorreyPines/FastInterpolations.jl/commit/d2ac3f72f5cce6da3a5791906ecf2fadda6f6e39"
        },
        "date": 1767594751276,
        "tool": "julia",
        "benches": [
          {
            "name": "construct/cubic_g10",
            "value": 1030.090909090909,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1664\nallocs=22\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":11,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "construct/linear_g1000",
            "value": 12.775775775775776,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":999,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "construct/linear_g10",
            "value": 12.775775775775776,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":999,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "construct/linear_g100",
            "value": 12.775775775775776,
            "unit": "ns",
            "extra": "gctime=0\nmemory=0\nallocs=0\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":999,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "construct/cubic_g100",
            "value": 3422.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=7744\nallocs=22\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":8,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "construct/cubic_g1000",
            "value": 28864,
            "unit": "ns",
            "extra": "gctime=0\nmemory=65200\nallocs=30\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/cubic_q10000",
            "value": 86300,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/linear_q10000",
            "value": 44002,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/cubic_q100",
            "value": 2295.4444444444443,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":9,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/cubic_q1",
            "value": 1413.6,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":10,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/linear_q1",
            "value": 39.54843592330979,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":991,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "oneshot/linear_q100",
            "value": 478.69897959183675,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":196,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/cubic_q10000",
            "value": 89757,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/linear_q10000",
            "value": 44012,
            "unit": "ns",
            "extra": "gctime=0\nmemory=80072\nallocs=3\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/cubic_q100",
            "value": 949.4615384615385,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":26,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/cubic_q1",
            "value": 51.48478701825558,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":986,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/linear_q1",
            "value": 49.262145748987855,
            "unit": "ns",
            "extra": "gctime=0\nmemory=64\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":988,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "eval/linear_q100",
            "value": 491.98974358974357,
            "unit": "ns",
            "extra": "gctime=0\nmemory=928\nallocs=2\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"evals_set\":false,\"samples\":10000,\"evals\":195,\"gcsample\":false,\"seconds\":10,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      }
    ]
  }
}