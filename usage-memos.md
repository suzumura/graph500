 使用法メモ
============

## シングルノード

* `mpirun -n <NumProcess> -output-filename <LogFileNamePrefix> -x OMP_NUM_THREADS=<NumThreads> ./runnable <Scale>`
    * NumProcess : プロセス数
    * LogFileNamePrefix : ログファイル名のプレフィクス(実際のログファイルは、ここで指定した文字列の後ろに`.1.<プロセスのインデックス>`を付けたものになる)
    * NumThreads : 1プロセス毎のスレッド数
    * Scale : Graph500 Benchmark における Scale の値
    * `mpirun -n 4 -output-filename ../l -x OMP_NUM_THREADS=1 ./runnable 20` のように使用


## マルチノード
