import matplotlib.pyplot as plt
import base64
from io import BytesIO

fig = plt.figure()
#plot sth

tmpfile = BytesIO()
fig.savefig(tmpfile, format='png')
encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

html = 'Some html head' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + 'Some more html'

with open('test.html','w') as f:
    f.write(html)