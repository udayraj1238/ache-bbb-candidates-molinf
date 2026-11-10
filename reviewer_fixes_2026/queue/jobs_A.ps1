# lane A: second and third share of the deeper local retrosynthesis (3000 iterations / 900 s, USPTO + ring-breaker) of the 162 extras that no earlier pass solved; lane B takes the first share.
# Resumable per molecule.  Idle priority.  The Vinardo exhaustiveness-32 docking is the last filler.
$E = "results\ext\"
$jobs = @(
  @{ name="M1_mol2mol_convert"; kind="py"; args=@("src\ad_v4_mol2mol_convert.py"); env=@{}; check={ (Test-Path "results\ext\mol2mol_raw.csv") }; tries=3; need_gb=1.0 },
  @{ name="M2_mol2mol_prep"; kind="py"; args=@("src\ad_v4_gan_baseline_prep.py","results\ext\mol2mol_raw.csv","results\ext\mol2mol_designs.csv"); env=@{}; check={ (Test-Path "results\ext\mol2mol_designs.csv") }; tries=4; need_gb=2.0; after="M1_mol2mol_convert" },
  @{ name="M3_mol2mol_gates"; kind="py"; args=@("src\ad_v4_gates.py","results\ext\gated_mol2mol","results\ext\mol2mol_designs.csv"); env=@{}; check={ (Test-Path "results\ext\gated_mol2mol_gated.csv") }; tries=4; need_gb=2.0; after="M2_mol2mol_prep" },
  @{ name="M4_mol2mol_report"; kind="py"; args=@("src\ad_v4_mol2mol_report.py"); env=@{}; check=$null; tries=2; need_gb=1.0; after="M3_mol2mol_gates" },
  @{ name="A5_retro_deeper_s1"; kind="retro"; args=@("src\ad_v2_retro.py","--in","${E}deep_local_todo3.csv","--out","${E}deep_local_retro3_s1.csv","--config","C:/retro/config_deeper_ring.yml","--skip-controls","--shard","1","--nshards","3"); env=@{RETRO_EXP="uspto,ringbreaker"}; check={ $m = [math]::Floor((Import-Csv "results\ext\deep_local_todo3.csv" | Measure-Object).Count / 3); (Test-Path "results\ext\deep_local_retro3_s1.csv") -and ((Import-Csv "results\ext\deep_local_retro3_s1.csv" | Measure-Object).Count -ge $m) }; tries=40; need_gb=3.0 },
  @{ name="A6_retro_deeper_s2"; kind="retro"; args=@("src\ad_v2_retro.py","--in","${E}deep_local_todo3.csv","--out","${E}deep_local_retro3_s2.csv","--config","C:/retro/config_deeper_ring.yml","--skip-controls","--shard","2","--nshards","3"); env=@{RETRO_EXP="uspto,ringbreaker"}; check={ $m = [math]::Floor((Import-Csv "results\ext\deep_local_todo3.csv" | Measure-Object).Count / 3); (Test-Path "results\ext\deep_local_retro3_s2.csv") -and ((Import-Csv "results\ext\deep_local_retro3_s2.csv" | Measure-Object).Count -ge $m) }; tries=40; need_gb=3.0 },
  @{ name="A7_report"; kind="py"; args=@("src\ad_v4_rescore_report.py"); env=@{}; check=$null; tries=2; need_gb=1.0 },
  @{ name="A8_vinardo_e32"; kind="py"; args=@("src\ad_v4_rescore_local.py","vinardo_e32","32","2"); env=@{VINA_SCORING="vinardo"}; check={ (Import-Csv "results\ext\rescore_vinardo_e32.csv" | Measure-Object).Count -ge 775 }; tries=8; need_gb=1.0 },
  @{ name="A9_report"; kind="py"; args=@("src\ad_v4_rescore_report.py"); env=@{}; check=$null; tries=2; need_gb=1.0 }
)
$at_end = @( ,@("src\ad_v4_rescore_report.py") )


