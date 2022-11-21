import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: '%.4f' % x)

df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()


def dataframe_info(dataframe):
    print("----- Take a Look at the Dataset -----")
    print("/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/")
    print(f"First 10 Observations: \n{dataframe.head(10)}")
    print("/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/")
    print(f"Last 10 Observations: \n{dataframe.tail(10)}")
    print("/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/")
    print(f"Dataframe Columns: \n{dataframe.columns}")
    print("/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/")
    print(f"Descriptive Statistics: \n{dataframe.describe().T}")
    print("/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/")
    print(f"NaN: \n{dataframe.isnull().sum()}")
    print("/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/")
    print(f"Variable Types: \n{dataframe.dtypes}")
    print("/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/")
    print(f"Number of Observations: \n{dataframe.shape[0]}")
    print(f"Number of Variables: \n{dataframe.shape[1]}")


dataframe_info(df)

df["total_transaction"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_price"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df.groupby(by="order_channel").agg({"master_id": "count",
                                    "TotalTransaction": ["sum", "mean"],
                                    "TotalPrice": ["sum", "mean"]})

df.groupby(by="master_id").agg({"TotalPrice": "sum"}).sort_values(by="TotalPrice", ascending=False).head(10)

df.groupby(by="master_id").agg({"TotalTransaction": "sum"}).sort_values(by="TotalTransaction", ascending=False).head(10)

df["last_order_date"].max()
analysis_date = dt.datetime(2021, 6, 1)

rfm = df.groupby(by="master_id").agg({"last_order_date": lambda x: (analysis_date - x.max()).days,
                                      "total_transaction": lambda x: x,
                                      "total_price": lambda x: x})

rfm.columns = ["recency", "frequency", "monetary"]

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm["RF_Score"] = (rfm["recency_score"].astype(str) +
                   rfm["frequency_score"].astype(str))

seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_Risk",
    r"[1-2]5": "cant_lose",
    r"3[1-2]": "about_to_sleep",
    r"33": "need_attention",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalist",
    r"5[4-5]": "champions"
}

rfm["segment"] = rfm["RF_Score"].replace(seg_map, regex=True)

rfm[["segment", "recency", "frequency", "monetary"]].groupby(by="segment").agg(["mean", "count"])

segments_counts = rfm['segment'].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots()

bars = ax.barh(range(len(segments_counts)),
              segments_counts,
              color='silver')
ax.set_frame_on(False)
ax.tick_params(left=False,
               bottom=False,
               labelbottom=False)
ax.set_yticks(range(len(segments_counts)))
ax.set_yticklabels(segments_counts.index)

for i, bar in enumerate(bars):
        value = bar.get_width()
        if segments_counts.index[i] in ['Can\'t loose']:
            bar.set_color('firebrick')
        ax.text(value,
                bar.get_y() + bar.get_height()/2,
                '{:,} ({:}%)'.format(int(value),
                                     int(value*100/segments_counts.sum())),
                va='center',
                ha='left')

plt.show()

rfm = rfm.reset_index()

target_segments_customer_ids = rfm[rfm["segment"].isin(["champions", "loyal_customers"])]["master_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("new_brand_target_customer_ids.csv", index=False)

target_segments_customer_ids = rfm[rfm["segment"].isin(["cant_loose", "hibernating", "new_customers"])]["master_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & ((df["interested_in_categories_12"].str.contains("ERKEK")) | (df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
cust_ids.to_csv("sale_target_customers_ids.csv", index=False)