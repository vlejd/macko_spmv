bits_per_value=$1
[[ -z "$bits_per_value" ]] && { echo "Parameter bits_per_value is empty" ; exit 1; }
echo "Running with ${bits_per_value} bits per value"
algos=$2
[[ -z "$algos" ]] && { echo "Parameter algos is empty" ; exit 1; }
echo "Benchmarking algorithms: ${algos}"


gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n1 | sed -E 's/.*: (.*)/\1/' | tr ' ' '_')

base_results=results_${bits_per_value}bit/${gpu_name}

mkdir -p ${base_results}/partial
complete_result_file="${base_results}/results.txt"

echo ${gpu_name} | tee ${complete_result_file}

Rs=(4096 8192 8192  32000 32000 28672 5120 5120  3584  4096  13824 18944 14336 4096  8192  11008 32000 20480 3584  21504 7168 28672 7168  27648 9216 36864 9216  36864 12288 49152 12288)
Cs=(4096 8192 29568 5120  8192  8192  5120 13824 20480 11008 5120  3584  4096  14336 28672 4096  4096  3584  18944 7168  7168 7168  28672 9216  9216 9216  36864 12288 12288 12288 49152)


for ((i=0; i<${#Rs[@]}; i++)); do
    R=${Rs[i]}
    C=${Cs[i]}

    for algo in ${algos};
    do
        for d in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05;
        do
            partial_result_file=${base_results}/partial/${R}_${C}_${algo}_${d}.txt
            ./benchmark 16 $R $C 100 1000 1 0 $algo 47 $d > $partial_result_file
            if grep -q "Final results:" "$partial_result_file"; then
                grep "Final results:" "$partial_result_file"
            else
                # Probably OOM
                echo "Final results:	$bits_per_value	$R	$C	$algo	$d	 	 	 "
            fi

        done
    done
done | tee -a ${complete_result_file}
