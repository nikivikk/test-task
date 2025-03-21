import base64
import pandas as pd
import numpy as np

from data_processor import Processor

from test_task.data import applications_64, data_64, output_64
from test_task.request import request

applications = pd.DataFrame(eval(base64.b64decode(applications_64)))
data = pd.DataFrame(eval(base64.b64decode(data_64)))
output = pd.DataFrame(eval(base64.b64decode(output_64)))

processor = Processor(applications=applications, data=data)
new_output = processor.get_data(request)

print(output)
