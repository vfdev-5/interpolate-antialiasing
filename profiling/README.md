# Call-stack profiling


## Install perf and FlameGraph

On Ubuntu 20.04 docker image, linux kernel is still 4.15, but perf and other tools are for the kernel 5.4.

```bash
uname -a
echo "deb http://archive.ubuntu.com/ubuntu/ bionic main universe\n" >> /etc/apt/sources.list
apt-get update
apt-get install -y linux-tools-4.15.0-20-generic linux-tools-4.15.0-20 linux-tools-4.15.0-20-lowlatency

rm -rf /usr/bin/perf
ln -s /usr/lib/linux-tools-4.15.0-20/perf /usr/bin/perf
```

```
git clone https://github.com/brendangregg/FlameGraph
```


## Usage

```bash
cmake -DTORCH_DIR=/pytorch/torch -DSTEP=step_two ..
make
```

```bash
OMP_NUM_THREADS=1 perf record -F 10000 --call-graph dwarf -- ./test
perf script > out.perf
../FlameGraph/stackcollapse-perf.pl out.perf > out.folded
../FlameGraph/flamegraph.pl out.folded > out.svg
```