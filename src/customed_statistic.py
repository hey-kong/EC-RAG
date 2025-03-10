class CustomedStatistic:
    data = {}
    print_list = False
    def init(self, args):
        self.print_list = args.detailed_logging
    def add(self, key: str, value):
        self.data[key] = value
    def add_to_list(self, key: str, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)
    
    def _summarize_statistics(self):
        """求均值"""
        avg_data = {}
        if self.data:
            for key, value in self.data.items():
                if isinstance(value, list):
                    # 如果list中是数字，求均值
                    if all(isinstance(i, (int, float)) for i in value):
                        avg = sum(value) / len(value)
                        avg_data[f'{key}_avg'] = avg
        self.data.update(avg_data)

    def _print_dict(self, input_dict):
        if input_dict:
            for key, value in input_dict.items():
                if isinstance(value, list):
                    if not self.print_list:
                        continue     # 跳过列表类型

                    # 列表类型，逐项缩进显示
                    print(f"  {key}:")
                    for item in value:
                        print(f"    - {item}")
                else:
                    print(f"  {key}: {value}")
    def dump(self):
        """打印统计结果"""
        self._summarize_statistics()

        # print("Arguments:")
        # args_dict = vars(self.args)
        # self._print_dict(args_dict)
        print("Statistics:")
        self._print_dict(self.data)

global_statistic = CustomedStatistic()