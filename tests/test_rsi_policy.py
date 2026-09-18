from rsi.policy import ExplorationPolicy, InitialParallelRefine

GRID = {"dim": [32, 64], "layers": [1, 2]}


def test_round_zero_covers_grid_deterministically():
    p = InitialParallelRefine(GRID, seed=7)
    cfgs = p.select_next_configs([], 4)
    assert len(cfgs) == 4
    assert p.select_next_configs([], 4) == cfgs


def test_refine_exploits_best():
    p = InitialParallelRefine({"lr": [0.1, 0.01, 0.001]}, seed=1,
                              top_k=1, random_frac=0.0)
    hist = [
        {"config": {"lr": 0.1}, "composite": 1.0, "status": "ok",
         "cost": {"gpu_hours": 0.1}},
        {"config": {"lr": 0.001}, "composite": 5.0, "status": "ok",
         "cost": {"gpu_hours": 0.1}},
    ]
    cfgs = p.select_next_configs(hist, 4)
    assert all(abs(c["lr"] - 0.001) <= 0.001 for c in cfgs)


def test_stops_and_degrees():
    p = InitialParallelRefine(GRID, seed=1)
    assert p.parallel_degree(3.7) == 3
    assert p.should_stop([], 10.0) is True
    assert p.should_stop([], 0.0) is False
