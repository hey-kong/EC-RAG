import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("log_file_path", help="path of log file")
args = parser.parse_args()
log_file_path = args.log_file_path
filename = log_file_path.split("/")[-1]

with open(log_file_path, "r", encoding="utf-8") as file:
    log_content = file.read()

rag_avg_time_match = re.search(r'rag_time_avg:\s*([\d.]+)', log_content)
average_f1_match = re.search(r"Overall F1: ([0-9.]+) \((\d+)\) \| Edge F1: ([0-9.]+) \((\d+)\) \| Cloud F1: ([0-9.]+) \((\d+)\)",log_content)
cloud_api_cost_avg_match = re.search(r'cloud_api_cost_avg:\s*([\d.]+(?:e[+-]?\d+)?)', log_content)
# cloud_api_cost_avg = 8.523844221105529e-05 有可能出现这种情况

if rag_avg_time_match and average_f1_match:
    rag_avg_time = rag_avg_time_match.group(1)
    average_f1, overall_num = float(average_f1_match.group(1)), int(average_f1_match.group(2))
    edge_f1, edge_num = float(average_f1_match.group(3)), int(average_f1_match.group(4))
    cloud_f1, cloud_num = float(average_f1_match.group(5)), int(average_f1_match.group(6))
    if cloud_api_cost_avg_match:
        cloud_api_cost_avg = cloud_api_cost_avg_match.group(1)
        print(f"{filename}\t rag_time_avg: {rag_avg_time}, \tavg_f1: {average_f1} ({overall_num}), \tedge_f1: {edge_f1} ({edge_num}), \tcloud_f1: {cloud_f1} ({cloud_num}), \tcloud_api_cost_avg: {cloud_api_cost_avg}")
    else:
        print(f"{filename}\t rag_time_avg: {rag_avg_time}, \tavg_f1: {average_f1} ({overall_num}), \tedge_f1: {edge_f1} ({edge_num}), \tcloud_f1: {cloud_f1} ({cloud_num})")
else:
    print("未找到匹配的数据")
