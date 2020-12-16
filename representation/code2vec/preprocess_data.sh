
DATA_DIR="../../data"
DATASET_NAME="test_case"
DATA_FILE="test_case.pkl"
PYTHON=python3

echo "Extracting files from data."
${PYTHON} ${DATA_DIR}/data2files.py --data_file ${DATA_DIR}/${DATA_FILE} --file_extension ".java" --output_dir ${DATA_DIR}/${DATASET_NAME}
echo "Finished files paths from data"