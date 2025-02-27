# 解析命令行参数
while getopts ":d:t:m:" opt; do
  case ${opt} in
    d)
      dp=$OPTARG
      ;;
    t)
      tp=$OPTARG
      ;;
    m)
      model_name=$OPTARG
      ;;
    \?)
      echo "无效选项: -$OPTARG" 1>&2
      exit 1
      ;;
    :)
      echo "选项 -$OPTARG 需要一个参数." 1>&2
      exit 1
      ;;
  esac
done

echo '==========================='
echo $dp
echo $tp
echo $model_name
echo '==========================='

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

/cpfs01/user/xingshuhao.dispatch/zyzeng/rllm/bin/python -m sglang.launch_server --host 0.0.0.0 --model-path ${model_name} --port 30000 --tp ${tp} --dp ${dp} --mem-fraction-static 0.8 --chunked-prefill-size 4096
# --schedule-conservativeness 1.2 --enable-torch-compile --mem-fraction-static 0.9 --disable-cuda-graph
# launch_server.py: error: unrecognized arguments: --max_prefill_tokens 32768
