import pandas as pd
from visualization.chart_generator import auto_chart

df = pd.DataFrame({
    "Category": ["A","B","C"],
    "Sales": [100,200,150]
})

fig = auto_chart(df)

fig.show()