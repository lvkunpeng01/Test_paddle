#!/bin/bash

MODE=$1
MODEL_LIST_FILE=$2

################################################### 可配置环境变量 #####################################################
# MEM_SIZE: 显存大小，默认值16G，设置示例：export MEM_SIZE=16
# DEVICE_TYPE: 设备类型，默认gpu，只支持小写，设置示例：export DEVICE_TYPE=gpu
# DEVICE_ID: 使用卡号，默认4卡，设置示例：export DEVICE_ID='0,1,2,3'
# TEST_RANGE: 测试范围，默认为空，设置示例：export TEST_RANGE='inference'
# MD_NUM: PR中MD文件改动数量，用于判断是否需要进行文档超链接检测，默认为空，设置示例：export MD_NUM=10
# WITHOUT_MD_NUM: PR中MD文件改动之外的改动文件数量，用于判断进行文档超链接检测后是否进行正常的CI，默认为空，设置示例：export WITHOUT_MD_NUM=10

if [[ $MODE == 'PaddleX' ]];then
    # set -x
    failed_cmd_list=""
else
    # set -x
    set +e
fi
successed_cmd_list=""


#################################################### Functions ######################################################

function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}

function func_parser_dataset_url(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[2]}
    echo ${tmp}
}

function get_device_list(){
    id_list=$DEVICE_ID
    if [[ $suite_name == "PaddleTS" || $mode == "predict" || $model_name == "YOWO" ]];then
        id_list=$FIRST_ID
    fi
    echo ${DEVICE_TYPE}:$id_list
}

# 运行命令并输出结果，PR级CI失败会重跑3次并异常退出，增量级和全量级会记录失败命令，最后打印失败的命令并异常退出
function run_command(){
    command=$1
    module_name_=$2
    time_stamp=$(date +"%Y-%m-%d %H:%M:%S")
    command="timeout 30m ${command}"
    printf "\e[32m|%-20s| %-50s | %-20s\n\e[0m" "[${time_stamp}]" "${command}"
    if [[ -z $CI_DEBUG ]];then
        sync
        echo 1 > /proc/sys/vm/drop_caches
        eval $command
    else
        echo $command
    fi
    last_status=${PIPESTATUS[0]}
    n=1
    retry_time=3
    # Try 2 times to run command if it fails
    while [[ $last_status != 0 ]]; do
        if [[ $last_status == 137 ]];then
            echo "CI 因内存资源耗尽而中断，将于90秒后自动重试..."
            retry_time=6
            sleep 90
            sync
            echo 1 > /proc/sys/vm/drop_caches
        else
            sleep 10
        fi
        n=`expr $n + 1`
        printf "\e[32m|%-20s| %-50s | %-20s\n\e[0m" "[${time_stamp}]" "${command}"
        eval $command
        last_status=${PIPESTATUS[0]}
        if [[ $n -ge $retry_time && $last_status != 0 ]]; then
            if [[ $last_status == 137 ]];then
                echo "CI 因内存资源耗尽而退出，如果在log中存在Kill字样，建议等待一段时间后重跑,如果连续重跑失败，请联系CI负责人排查问题。"
            fi
            if [[ $MODE != 'PaddleX' ]];then
                echo "Retry $retry_time times failed with command: ${command}"
                exit 1
            else
                break
            fi
        fi
    done
    if [[ $MODE == 'PaddleX' ]];then
        if [[ $last_status != 0 ]]; then
            failed_cmd_list="$failed_cmd_list\n${command}"
            echo "CI_FAILED_CMD: ${command}"
        else
            successed_cmd_list="$successed_cmd_list\n${command}"
        fi
    fi
}

# 准备数据集并做数据校验
function prepare_dataset(){
    if [[ $dataset_url == 'https:' ]];then
        train_data_file=""
        return
    fi
    download_dataset_cmd="${PYTHON_PATH} ${BASE_PATH}/tools.py --download_dataset --config_path ${check_dataset_yaml} --dataset_url ${dataset_url}"
    run_command ${download_dataset_cmd} ${module_name}
    model_output_path=${MODULE_OUTPUT_PATH}/${module_name}_dataset_check
    check_dataset_cmd="${PYTHON_PATH} main.py -c ${check_dataset_yaml} -o Global.mode=check_dataset -o Global.output=${model_output_path} "
    run_command ${check_dataset_cmd} ${module_name}
    checker_cmd="${PYTHON_PATH} ${BASE_PATH}/tools.py --check --check_dataset_result --output ${model_output_path} --module_name ${module_name}"
    run_command ${checker_cmd} ${module_name}
    dataset_dir=`cat $check_dataset_yaml | grep  -m 1 dataset_dir | awk  {'print$NF'}| sed 's/"//g'`
    if [[ ! -z $train_list_name ]]; then
        train_data_file=${dataset_dir}/${train_list_name}
        mv $train_data_file $train_data_file.bak
    fi
}

# 对给定的模型列表运行模型相应的train和evaluate等操作
function run_models(){
    config_files=$1
    for config_path in $config_files;do
        config_path=$(func_parser_value "${config_path}")
        batch_size=`cat $config_path | grep  -m 1 batch_size | awk  {'print$NF'}`
        device=`cat $config_path | grep  -m 1 device | awk  {'print$NF'}`
        IFS=$','
        device_list=(${device})
        device_num=${#device_list[@]}
        IFS=$' '
        # 根据内存大小调整batch_size
        if [[ $MEM_SIZE -lt 16 ]];then
            if [[ $batch_size -ge 4 ]];then
                batch_size=`expr $batch_size / 4`
            else
                batch_size=1
            fi
        elif [[ $MEM_SIZE -lt 32 ]];then
            if [[ $batch_size -ge 2 ]];then
                batch_size=`expr $batch_size / 2`
            else
                batch_size=1
            fi
        fi
        # 根据batch_size和device_num调整数据集的数量
        data_num=`expr $device_num \* $batch_size`
        if [[ ! -z $train_data_file ]]; then
            if [[ $module_name == ts* ]]; then
                data_num=`expr $device_num \* $batch_size \* 30`
                data_num=`expr $data_num + 1`
            fi
            head -n $data_num $train_data_file.bak > $train_data_file
        fi
        yaml_name=${config_path##*/}
        model_name=${yaml_name%.*}
        model_list="${model_list} ${model_name}"
        model_output_path=${MODULE_OUTPUT_PATH}/${module_name}_output/${model_name}
        evaluate_weight_path=${model_output_path}/${best_weight_path}
        if [[ $inference_model_dir == 'null' ]];then
            inference_weight_path="None"
        else
            inference_weight_path=${model_output_path}/${inference_model_dir}
        fi
        mkdir -p $model_output_path
        IFS=$'|'
        check_flag=""
        run_model_list=(${run_model})
        #### 有新推理接入，使用新推理 ####
        if [[ $USE_NEW_INFERENCE == 1 ]];then
            set_flag=`ls paddlex/inference/models_new | grep ${module_name}`
            if [[ -z $set_flag ]];then
                export PADDLE_PDX_NEW_PREDICTOR=0
            else
                export PADDLE_PDX_NEW_PREDICTOR=1
            fi
        fi
        for mode in ${run_model_list[@]};do
            black_model=`eval echo '$'"${mode}_black_list"|grep "^${model_name}$"`
            if [[ ! -z $black_model ]];then
                # 黑名单模型，不运行
                if [[ $mode == "train" ]];then
                    check_flag="False"
                fi
                echo "$model_name is in ${mode}_black_list, so skip it."
                continue
            fi
            device_info=$(get_device_list)
            # TEST_RANGE 为空时为普通全流程测试
            if [[ -z $TEST_RANGE ]];then
                # 适配导出模型时，需要指定输出路径
                if [[ $model_name == PP-YOLOE_plus-M || $model_name == PP-YOLOE_plus-S || $model_name == PP-YOLOE_plus-X ]];then
                    epochs_iters=5
                elif [[ $model_name == SOLOv2 ]];then
                    epochs_iters=10
                fi
                base_mode_cmd="${PYTHON_PATH} main.py -c ${config_path} -o Global.mode=${mode} -o Global.device=${device_info} -o Train.epochs_iters=${epochs_iters} -o Train.batch_size=${batch_size} -o Evaluate.weight_path=${evaluate_weight_path} -o Predict.model_dir=${inference_weight_path}"
                if [[ $mode == "export" ]];then
                    model_export_output_path=${model_output_path}/export
                    mkdir -p $model_export_output_path
                    weight_dict[$model_name]="$model_export_output_path"
                    run_mode_cmd="${base_mode_cmd} -o Global.output=${model_export_output_path}"
                else
                    run_mode_cmd="${base_mode_cmd} -o Global.output=${model_output_path}"
                fi
                run_command ${run_mode_cmd} ${module_name}
            # TEST_RANGE 为inference时，只测试官方模型预测
            elif [[ $TEST_RANGE == "inference" && $mode == "predict" ]];then
                offcial_model_predict_cmd="${PYTHON_PATH} main.py -c ${config_path} -o Global.mode=predict -o Predict.model_dir=None -o Global.output=${model_output_path}_offical_predict -o Global.device=${device_info}"
                run_command ${offcial_model_predict_cmd} ${module_name}
                continue
            # TEST_RANGE 为其他非空值时，不做模型级别测试
            else
                continue
            fi
        done
        if [[ ! -z $check_flag ]];then
            # 黑名单模型，不做检查
            echo "$model_name is in ${mode}_black_list, so skip it."
            continue
        fi
        if [[ $check_options == 'null' ]]; then
            continue
        fi
        # runing_train为空时为普通全过程测试
        if [[ -z $TEST_RANGE ]];then
            check_options_list=(${check_options})
            rm_output=""
            for check_option in ${check_options_list[@]};do
                # 运行产出检查脚本
                checker_cmd="${PYTHON_PATH} ${BASE_PATH}/tools.py --check --$check_option --output ${model_output_path} --check_weights_items ${check_weights_items} --module_name ${module_name} 2>&1 | tee ${model_output_path}/checker_${check_option}.log"
                run_command ${checker_cmd} ${module_name}
                if [[ $last_status != 0 ]];then
                    rm_output="${rm_output} false"
                fi
            done
            if [[ -z $rm_output ]];then
                rm -rf ${model_output_path}
            fi
        fi
    done
    model_dict[$module_name]="$model_list"
}

# 准备数据集和运行模型
function prepare_and_run(){
    config_file_list=$1
    if [[ ! -z $config_file_list ]];then
        # runing_train为空时为普通全过程测试,非空时代表无训练测试，无需准备数据集
        if [[ -z $TEST_RANGE ]];then
            prepare_dataset
        fi
        run_models "${config_file_list}"
    fi
}


#################################################### PaddleX CI ######################################################
# PaddleX CI 区分为 PR 级、套件级、全量级和增量级
# PR 级：运行 pr_list.txt 中的重点模型列表，示例 bash ci_run.sh
# 套件级：运行 pr_list.txt 中指定套件的部分模型，示例 bash ci_run.sh PaddleClas
# 全量级：根据 config.txt 抓取 PaddleX 支持的所有模型并测试，示例 bash ci_run.sh PaddleX
# 增量级：根据 config.txt 和 传入的 changed_list.txt 抓取新增或修改的模型，对新增模型进行增量测试，示例 bash ci_run.sh PaddleX changed_list.txt

# 指定 python
PYTHON_PATH="python"
# 获取当前脚本的绝对路径，获得基准目录
BASE_PATH=$(cd "$(dirname $0)"; pwd)
MODULE_OUTPUT_PATH=${BASE_PATH}/outputs
CONFIG_FILE=${BASE_PATH}/config.txt
echo $CI_DEBUG
if [[ -z $CI_DEBUG ]];then
    pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
    pip install beautifulsoup4==4.12.3
    pip install tqdm
    pip install markdown
    pip install prettytable
    pip install colorlog
fi
declare -A weight_dict
declare -A model_dict

#################################################### 代码风格检查 ######################################################
if [[ $DEVICE_TYPE == 'gpu' ]];then
    pre-commit
    last_status=${PIPESTATUS[0]}
    if [[ $last_status != 0 ]]; then
        echo "pre-commit check failed, please fix it first."
        exit 1
    fi
fi

#################################################### 文档超链接检查 ######################################################
if [[ ! $MD_NUM -eq 0 && ! -z $MD_NUM  ]];then
    checker_url_cmd="${PYTHON_PATH} ${BASE_PATH}/tools.py --check_url -m internal"
    eval $checker_url_cmd
    last_status=${PIPESTATUS[0]}
    if [[ $last_status != 0 ]]; then
        echo "check urls in documentation failed. please fix invalid urls."
        exit 1
    elif [[ $WITHOUT_MD_NUM -eq 0 ]];then
        echo "this pr is all markdown files, so skip it."
        exit 0
    fi
fi

# 安装paddlex，完成环境准备
install_pdx_cmd="pip install -e ."

if [[ -z $CI_DEBUG ]];then
    eval $install_pdx_cmd
    last_status=${PIPESTATUS[0]}
    if [[ $last_status != 0 ]]; then
        if [[ $last_status == 137 ]];then
            echo "CI 因内存资源耗尽而退出，如果在log中存在Kill字样，建议等待一段时间后重跑,如果连续重跑失败，请联系CI负责人排查问题。"
        else
            echo "install paddlex failed, please fix it first."
        fi
        exit 1
    fi
fi

if [[ -z $MEM_SIZE ]]; then
    MEM_SIZE=32
fi

if [[ -z $DEVICE_TYPE ]]; then
    DEVICE_TYPE='gpu'
fi

PIPE_TYPE=$DEVICE_TYPE
if [[ $DEVICE_TYPE == 'dcu' ]]; then
    DEVICE_TYPE='gpu'
fi

if [[ -z $DEVICE_ID ]]; then
    DEVICE_ID='0,1,2,3'
fi
FIRST_ID=`echo $DEVICE_ID | awk -F ',' {'print$1'}`

if [[ -z $MODE ]]; then
    install_deps_cmd="paddlex --install -y"
elif [[ $MODE == "PaddleX" ]];then
    install_deps_cmd="paddlex --install -y"
else
    install_deps_cmd="paddlex --install --use_local_repos $MODE"
fi

# # 只测试产线推理无需安装套件库
if [[ $TEST_RANGE != "pipeline" || -z $CI_DEBUG ]];then
    eval ${install_deps_cmd}
    last_status=${PIPESTATUS[0]}
    if [[ $last_status != 0 ]]; then
        if [[ $last_status == 137 ]];then
            echo "CI 因内存资源耗尽而退出，如果在log中存在Kill字样，建议等待一段时间后重跑,如果连续重跑失败，请联系CI负责人排查问题。"
        else
            echo "install suite repos failed, please fix it first."
        fi
        exit 1
    fi
fi
pip freeze > all_packages.txt

cmd="${PYTHON_PATH} ${BASE_PATH}/tools.py --check_env"
eval $cmd
last_status=${PIPESTATUS[0]}
if [[ $last_status != 0 ]]; then
    echo "check env failed, please fix it first."
    exit 1
fi

#################################################### 模型级测试 ######################################################
IFS=$' '
black_list_file=${BASE_PATH}/black_list.txt
set +e
all_black_list=`cat ${black_list_file} | grep All: | awk -F : {'print$2'}`
train_black_list=`cat ${black_list_file} | grep Train: | awk -F : {'print$2'}`
train_black_list="$all_black_list
$train_black_list"
evaluate_black_list=`cat ${black_list_file} | grep Evaluate: | awk -F : {'print$2'}`
evaluate_black_list="$all_black_list
$evaluate_black_list"
predict_black_list=`cat ${black_list_file} | grep Predict: | awk -F : {'print$2'}`
predict_black_list="$all_black_list
$predict_black_list"
export_black_list=`cat ${black_list_file} | grep Export: | awk -F : {'print$2'}`
export_black_list="$all_black_list
$export_black_list"
pipeline_black_list=`cat ${black_list_file} | grep Pipeline: | awk -F : {'print$2'}`
echo "======================= 黑名单信息 ===========================
##############train_black_list###############
$train_black_list
##############evaluate_black_list###############
$evaluate_black_list
##############predict_black_list###############
$predict_black_list
##############export_black_list###############
$export_black_list
##############pipeline_black_list###############
$pipeline_black_list
==============================================================="

echo "============================ 开始模型级CI测试 ====================================="

IFS='*'
modules_info_list=($(cat ${CONFIG_FILE}))
all_module_names=`cat $CONFIG_FILE | grep module_name | awk -F ':' {'print$2'}`

unset http_proxy https_proxy no_proxy
IFS=$' '
for modules_info in ${modules_info_list[@]}; do
    IFS='='
    model_list=''
    info_list=($modules_info)
    for module_info in ${info_list[@]}; do
        if [[ $module_info == *check_dataset_yaml* ]]; then
            # 数据准备，获取模型信息和运行模式
            IFS=$'\n'
            lines=(${module_info})
            line_num=0
            suite_name=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            module_name=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            check_dataset_yaml=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            dataset_url=https:$(func_parser_dataset_url "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            train_list_name=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            run_model=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            check_options=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            check_weights_items=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            best_weight_path=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            inference_model_dir=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            epochs_iters=$(func_parser_value "${lines[line_num]}")
            if [[ $MODE == "PaddleX" ]];then
                if [[ ! -z $MODEL_LIST_FILE ]];then
                    new_model_info=`cat $MODEL_LIST_FILE`
                    new_model_module_names=`cat $MODEL_LIST_FILE | awk -F '/' {'print$4'} | sort -u`
                    for new_module_info in ${new_model_module_names[@]};do
                        module=`echo "${all_module_names[@]}" | grep $new_module_info`
                        if [[ -z $module ]];then
                            echo "检测到未支持的module类型: $new_module_info ，请准备对应模块的测试数据集（为保障测试时间，测试数据集越小越好），并联系CI添加该模块的相关配置后重跑CI！"
                            exit 1
                        fi
                        if [[ $new_module_info == $module_name ]];then
                            module_info=`cat $MODEL_LIST_FILE | grep $module_name | xargs -n1 -I {} echo config_path:{}`
                            echo $module_info
                            prepare_and_run "${module_info}"
                        fi
                    done
                else
                    module_info=`ls paddlex/configs/modules/${module_name} | xargs -n1 -I {} echo config_path:paddlex/configs/modules/${module_name}/{}`
                    prepare_and_run "${module_info}"
                fi
                continue
            elif [[ $MODE == $suite_name ]];then
                module_info=`cat ${BASE_PATH}/pr_list.txt | grep -v "^#" | grep paddlex/configs/modules/${module_name}`
                prepare_and_run "${module_info}"
            elif [[ -z $MODE ]];then
                module_info=`cat ${BASE_PATH}/pr_list.txt | grep -v "^#" | grep paddlex/configs/modules/${module_name}`
                prepare_and_run "${module_info}"
            else
                continue
            fi
        fi
    done
done

if [[ $PIPE_TYPE != 'gpu' ]];then
    exit 0
fi

if [[ $LOCAL_RUN_WO_PIPLINE == 1 ]];then
    if [[ $MODE == 'PaddleX' && ! -z $failed_cmd_list ]];then
        echo "以下为失败的命令列表："
        echo -e "$failed_cmd_list"
        echo "可在全量日志中检索“CI_FAILED_CMD”关键字，快速定位到运行失败命令位置，相关日志在关键字上方。如果有“Kill”相关字样，可以忽略问题重跑；如果是产出检查失败，建议继续向上排查，CI顺序为【数据->训练->评估->预测->导出】"
        exit 1
    fi
    exit 0
fi

if [[ ! -z $USE_NEW_INFERENCE ]];then
    export USE_NEW_INFERENCE=""
fi

echo "============================ 开始产线级CI测试 ====================================="

#################################################### 产线级测试 ######################################################
IFS=$'\n'
if [[ ! -z $USE_NEW_INFERENCE ]];then
    PIPELINE_YAML_LIST=`ls paddlex/configs/pipelines | grep .yaml`
else
    PIPELINE_YAML_LIST=`ls paddlex/pipelines | grep .yaml`
fi

function check_pipeline() {
	pipeline=$1
	model=$2
    model_dir=$3
	img=$4
    output_dir_name=`echo $model | sed 's/ /_/g'`
    output_path=${MODULE_OUTPUT_PATH}/pipeline_output/${output_dir_name}
	rm -rf $output_path
	mkdir -p $output_path
	cd $output_path
    cmd="timeout 30m paddlex --pipeline ${pipeline} --input ${img} --device ${DEVICE_TYPE}:${FIRST_ID} 2>&1 | tee ${BASE_PATH}/outputs/${pipeline}.log"
	echo $cmd
    if [[ -z $CI_DEBUG ]];then
        eval $cmd
    fi
    last_status=${PIPESTATUS[0]}
    n=1
    retry_time=3
    # Try 2 times to run command if it fails
    if [[ $MODE != 'PaddleX' ]];then
        while [[ $last_status != 0 ]]; do
            if [[ $last_status == 137 ]];then
                echo "CI 因内存资源耗尽而中断，将于90秒后自动重试..."
                retry_time=6
                sleep 90
                sync
                echo 1 > /proc/sys/vm/drop_caches
            fi
            sleep 10
            n=`expr $n + 1`
            printf "\e[32m|%-20s| %-50s | %-20s\n\e[0m" "[${time_stamp}]" "${cmd}"
            eval $cmd
            last_status=${PIPESTATUS[0]}
            if [[ $n -eq $retry_time && $last_status != 0 ]]; then
                if [[ $last_status == 137 ]];then
                    echo "CI 因内存资源耗尽而退出，如果在log中存在Kill字样，建议等待一段时间后重跑,如果连续重跑失败，请联系CI负责人排查问题。"
                fi
                echo "Retry 2 times failed with command: ${cmd}"
                exit 1
            fi
        done
    elif [[ $last_status != 0 ]]; then
        failed_cmd_list="$failed_cmd_list\n${cmd}"
        echo "Run ${cmd} failed"
    else
        successed_cmd_list="$successed_cmd_list\n${cmd}"
    fi
	cd -
}

for pipeline_yaml in ${PIPELINE_YAML_LIST[@]};do
    if [[ ! -z $USE_NEW_INFERENCE ]];then
        pipeline_name=`cat paddlex/configs/pipelines/${pipeline_yaml} | grep pipeline_name | awk {'print$2'} | head -n 1`
    else
        pipeline_name=`cat paddlex/pipelines/${pipeline_yaml} | grep pipeline_name | awk {'print$2'}`
    fi
    IFS=' '
    black_pipeline=`echo ${pipeline_black_list}|grep "^${pipeline_name}$"`
    IFS=$'\n'
    if [[ ! -z $black_pipeline ]];then
        # 黑名单产线，不运行
        echo "$pipeline_name is in pipeline_black_list, so skip it."
        continue
    fi
    input=`cat ci/pipeline_config.txt | grep "${pipeline_name}: " | awk -F ": " {'print$2'}`
    check_pipeline $pipeline_name "" "" $input
done

if [[ $MODE == 'PaddleX' && -z $MODEL_LIST_FILE ]];then
    echo -e "${successed_cmd_list}" > ${BASE_PATH}/outputs/success_cmd_list.txt
    echo -e "${failed_cmd_list}" > ${BASE_PATH}/outputs/failed_cmd_list.txt
    save_result_cmd="${PYTHON_PATH} ${BASE_PATH}/tools.py --save_result --successed_cmd ${BASE_PATH}/outputs/success_cmd_list.txt --failed_cmd ${BASE_PATH}/outputs/failed_cmd_list.txt"
    echo $save_result_cmd
    eval $save_result_cmd
fi

if [[ $MODE == 'PaddleX' && ! -z $failed_cmd_list ]];then
    echo "以下为失败的命令列表："
    echo -e "$failed_cmd_list"
    echo "可在全量日志中检索“CI_FAILED_CMD”关键字，快速定位到运行失败命令位置，相关日志在关键字上方。如果有“Kill”相关字样，可以忽略问题重跑；如果是产出检查失败，建议继续向上排查，CI顺序为【数据->训练->评估->预测->导出】"
    exit 1
fi
