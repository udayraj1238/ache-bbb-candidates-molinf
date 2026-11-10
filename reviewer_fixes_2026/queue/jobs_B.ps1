# lane B: item 2 follow-through and routes.  Every step is resumable or cheap to repeat.  Idle priority.
$E = "results\ext\"
function Ex($f) { { Test-Path $f }.GetNewClosure() }
function Rows($f) { (Import-Csv $f | Measure-Object).Count }
$jobs = @(
  @{ name="B01_i2_collect"; kind="py"; args=@("src\ad_v4_extras.py","collect"); env=@{EXTRAS_TAG="3"}; check=(Ex "${E}extras3_in.csv"); tries=3; need_gb=2.0 },
  @{ name="B02_i2_stereo"; kind="py"; args=@("src\ad_v4_extras.py","stereo"); env=@{EXTRAS_TAG="3"}; check=(Ex "${E}extras3_stereo.csv"); tries=3; need_gb=2.0 },
  @{ name="B03_i2_offline"; kind="py"; args=@("src\ad_v4_extras.py","offline"); env=@{EXTRAS_TAG="3"}; check=(Ex "${E}extras3_offline0.csv"); tries=3; need_gb=2.0 },
  @{ name="B04_i2_novelty"; kind="py"; args=@("src\ad_v4_extras.py","novelty"); env=@{EXTRAS_TAG="3"}; check={ $f = "results\ext\extras3_novelty.csv"; (Test-Path $f) -and ((Import-Csv $f | Where-Object { $_.novelty_unresolved -eq "True" } | Measure-Object).Count -le 0.1 * (Import-Csv $f | Measure-Object).Count) }; tries=4; need_gb=1.5 },
  @{ name="B05_i2_dock"; kind="py"; args=@("src\ad_v4_extras.py","dock"); env=@{EXTRAS_TAG="3"}; check=(Ex "${E}extras3_final0.csv"); tries=6; need_gb=1.5 },
  @{ name="B06_i2_toretro"; kind="py"; args=@("src\ad_v4_extras.py","toretro"); env=@{EXTRAS_TAG="3"}; check=(Ex "${E}extras3_toretro.csv"); tries=3; need_gb=1.0 },
  @{ name="B07_i2_sim"; kind="py"; args=@("src\ad_v4_similarity_search.py","${E}extras3_toretro.csv","results\AD_V4_SIMILARITY_SEARCH_EXTRAS3.csv"); env=@{}; check={ (Test-Path "results\AD_V4_SIMILARITY_SEARCH_EXTRAS3.csv") -and ((Import-Csv "results\AD_V4_SIMILARITY_SEARCH_EXTRAS3.csv" | Select-Object -Unique smiles | Measure-Object).Count -ge (Import-Csv "results\ext\extras3_toretro.csv" | Measure-Object).Count) }; tries=6; need_gb=1.0 },
  @{ name="B08_i2_todo"; kind="py"; args=@("src\ad_v4_queue_todo.py","new"); env=@{}; check=(Ex "${E}deep_local_todo2.csv"); tries=2; need_gb=1.0 },
  @{ name="B09_i2_retro"; kind="retro"; args=@("src\ad_v2_retro.py","--in","${E}deep_local_todo2.csv","--out","${E}deep_local_retro2.csv","--config","C:/retro/config_deep_ring.yml","--skip-controls"); env=@{RETRO_EXP="uspto,ringbreaker"}; check={ $n = (Import-Csv "results\ext\deep_local_todo2.csv" | Measure-Object).Count; ($n -eq 0) -or ((Test-Path "results\ext\deep_local_retro2.csv") -and ((Import-Csv "results\ext\deep_local_retro2.csv" | Measure-Object).Count -ge $n)) }; tries=8; need_gb=3.0 },
  @{ name="B10_final1"; kind="py"; args=@("src\ad_v4_final_set.py"); env=@{}; check=$null; tries=2; need_gb=1.5 },
  @{ name="B11_i2b_parents"; kind="py"; args=@("src\ad_v4_queue_parents.py"); env=@{}; check=(Ex "${E}item2b_parents.csv"); tries=2; need_gb=1.0 },
  @{ name="B12_i2b_bbb"; kind="py"; args=@("src\memrun.py","1.4","src\design_disease_candidates.py","--disease","alzheimers","--parents-file","${E}item2b_parents.csv","--max-per-scaffold","100000","--out","${E}item2b_designs_bbb.csv"); env=@{}; check=(Ex "${E}item2b_designs_bbb.csv"); tries=4; need_gb=2.0 },
  @{ name="B13_i2b_b1"; kind="py"; args=@("src\memrun.py","1.4","src\design_disease_candidates.py","--disease","alzheimers","--parents-file","${E}item2b_parents.csv","--max-per-scaffold","100000","--library","${E}lib_B1_ache_preserving.csv","--library-mode","all","--out","${E}item2b_designs_b1.csv"); env=@{}; check=(Ex "${E}item2b_designs_b1.csv"); tries=4; need_gb=2.0 },
  @{ name="B14_i2b_gates"; kind="py"; args=@("src\ad_v4_gates.py","${E}item2b_gated","${E}item2b_designs_bbb.csv","${E}item2b_designs_b1.csv"); env=@{}; check=(Ex "${E}item2b_gated_gated.csv"); tries=4; need_gb=2.0 },
  @{ name="B15_i2b_collect"; kind="py"; args=@("src\ad_v4_extras.py","collect"); env=@{EXTRAS_TAG="4"}; check=(Ex "${E}extras4_in.csv"); tries=3; need_gb=2.0 },
  @{ name="B16_i2b_stereo"; kind="py"; args=@("src\ad_v4_extras.py","stereo"); env=@{EXTRAS_TAG="4"}; check=(Ex "${E}extras4_stereo.csv"); tries=3; need_gb=2.0 },
  @{ name="B17_i2b_offline"; kind="py"; args=@("src\ad_v4_extras.py","offline"); env=@{EXTRAS_TAG="4"}; check=(Ex "${E}extras4_offline0.csv"); tries=3; need_gb=2.0 },
  @{ name="B18_i2b_novelty"; kind="py"; args=@("src\ad_v4_extras.py","novelty"); env=@{EXTRAS_TAG="4"}; check={ $f = "results\ext\extras4_novelty.csv"; (Test-Path $f) -and ((Import-Csv $f | Where-Object { $_.novelty_unresolved -eq "True" } | Measure-Object).Count -le 0.1 * (Import-Csv $f | Measure-Object).Count) }; tries=4; need_gb=1.5 },
  @{ name="B19_i2b_dock"; after="B18_i2b_novelty"; kind="py"; args=@("src\ad_v4_extras.py","dock"); env=@{EXTRAS_TAG="4"}; check=(Ex "${E}extras4_final0.csv"); tries=6; need_gb=1.5 },
  @{ name="B20_i2b_toretro"; after="B19_i2b_dock"; kind="py"; args=@("src\ad_v4_extras.py","toretro"); env=@{EXTRAS_TAG="4"}; check=(Ex "${E}extras4_toretro.csv"); tries=3; need_gb=1.0 },
  @{ name="B21_i2b_sim"; after="B20_i2b_toretro"; kind="py"; args=@("src\ad_v4_similarity_search.py","${E}extras4_toretro.csv","results\AD_V4_SIMILARITY_SEARCH_EXTRAS4.csv"); env=@{}; check={ (Test-Path "results\AD_V4_SIMILARITY_SEARCH_EXTRAS4.csv") -and ((Import-Csv "results\AD_V4_SIMILARITY_SEARCH_EXTRAS4.csv" | Select-Object -Unique smiles | Measure-Object).Count -ge (Import-Csv "results\ext\extras4_toretro.csv" | Measure-Object).Count) }; tries=6; need_gb=1.0 },
  @{ name="B22_i2b_todo"; after="B20_i2b_toretro"; kind="py"; args=@("src\ad_v4_queue_todo.py","new4"); env=@{}; check=(Ex "${E}deep_local_todo4.csv"); tries=2; need_gb=1.0 },
  @{ name="B23_i2b_retro"; after="B22_i2b_todo"; kind="retro"; args=@("src\ad_v2_retro.py","--in","${E}deep_local_todo4.csv","--out","${E}deep_local_retro4.csv","--config","C:/retro/config_deep_ring.yml","--skip-controls"); env=@{RETRO_EXP="uspto,ringbreaker"}; check={ $n = (Import-Csv "results\ext\deep_local_todo4.csv" | Measure-Object).Count; ($n -eq 0) -or ((Test-Path "results\ext\deep_local_retro4.csv") -and ((Import-Csv "results\ext\deep_local_retro4.csv" | Measure-Object).Count -ge $n)) }; tries=8; need_gb=3.0 },
  @{ name="B24_final2"; after="B23_i2b_retro"; kind="py"; args=@("src\ad_v4_final_set.py"); env=@{}; check=$null; tries=2; need_gb=1.5 },
  @{ name="B26_retro_deeper_s0"; kind="retro"; args=@("src\ad_v2_retro.py","--in","${E}deep_local_todo3.csv","--out","${E}deep_local_retro3_s0.csv","--config","C:/retro/config_deeper_ring.yml","--skip-controls","--shard","0","--nshards","3"); env=@{RETRO_EXP="uspto,ringbreaker"}; check={ $m = [math]::Floor((Import-Csv "results\ext\deep_local_todo3.csv" | Measure-Object).Count / 3); (Test-Path "results\ext\deep_local_retro3_s0.csv") -and ((Import-Csv "results\ext\deep_local_retro3_s0.csv" | Measure-Object).Count -ge $m) }; tries=40; need_gb=3.0 },
  @{ name="B27_final3"; kind="py"; args=@("src\ad_v4_final_set.py"); env=@{}; check=$null; tries=2; need_gb=1.5 }
)
$at_end = @( ,@("src\ad_v4_final_set.py") )


