import configparser
import os



class Config:
    def __init__(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        self.project_dir = os.path.dirname(current_dir)
        config_path = os.path.join(self.project_dir, 'config/config.ini')
        self.config.read(config_path)

    def get_config(self, section, key):
        return self.config[section][key]

    def section(self, section):
        if section == 'General':
            d = dict(self.config.items(section))
            d['project_dir'] = self.project_dir
        elif section == 'Visualization':
            d = dict(self.config.items(section))
            d["figsize"] = eval(d["figsize"])
            for section in self.config.sections():
                if section.startswith('Visualization.font'):
                    d[section.split(".")[1]] = dict(self.config.items(section))
        elif section == 'Data':
            d = dict(self.config.items(section))
            d["sampling_frequency"] = int(d["sampling_frequency"])
            d["labels"] = eval(d["labels"])
        else:
            raise ValueError("Section not found in config file.")
        return d

config = Config()
data_config = config.section('Data')
general_config = config.section('General')
plot_config = config.section('Visualization')
