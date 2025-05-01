import io
import datetime
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── 1) Page setup & global font ──────────────────────────────────────────
st.set_page_config(page_title="Options P&L Summary", layout="wide")
st.markdown(
    """
    <style>
      html, body, [class*="css"] {
        font-family: 'Arial', sans-serif !important;
        background-color: #0E1117;
      }
      .centered {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 100px;
        margin-bottom: 100px;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ── 2) Landing / splash screen ─────────────────────────────────────────────
with st.container():
    st.markdown(
        """
        <div class="centered">
          <h1 style="text-align: center; color: white; line-height: 1.2;">
            Welcome to<br/>
            Options P&L Summary
          </h1>
          <p style="text-align: center; color: #888888; margin-top: 30px;">
            Upload Schwab Transaction Summary CSV file to Start
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# center the uploader under the splash
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded = st.file_uploader("", type="csv", accept_multiple_files=True)

# if no files yet, stop here
if not uploaded:
    st.stop()

# ── 3) Main title once files are in ────────────────────────────────────────
st.title("Options P&L Summary")

# ── 4) CSV parsing & cleaning ──────────────────────────────────────────────
def process(df):
    opts = df[df['Description'].str.startswith(('PUT','CALL'))].copy()
    parts = opts['Symbol'].str.split(' ', expand=True)
    opts['Underlying'] = parts[0]
    opts['Expiry']     = pd.to_datetime(parts[1])
    opts['Strike']     = parts[2].astype(float)
    opts['OptType']    = parts[3]
    opts['Price']      = opts['Price'].str.replace(r'[\$,]', '', regex=True).astype(float).fillna(0)
    opts['Amount']     = opts['Amount'].str.replace(r'[\$,]', '', regex=True).astype(float).fillna(0)
    opts['CashFlow']   = opts['Amount']
    opts['Quantity']   = pd.to_numeric(opts['Quantity'], errors='coerce').fillna(0).abs().astype(int)
    opts['ActionType'] = opts['Action']
    opts['Date']       = (
        opts['Date'].astype(str)
            .str.extract(r'(\d{1,2}/\d{1,2}/\d{4})')[0]
            .pipe(pd.to_datetime, format="%m/%d/%Y", errors='raise')
    )
    opts['Month'] = opts['Date'].dt.to_period('M').astype(str)
    return opts

df   = pd.concat([pd.read_csv(f) for f in uploaded], ignore_index=True)
opts = process(df)

# ── 5) Settings expander ───────────────────────────────────────────────────
with st.expander("Settings", expanded=False):
    tax_rate_pct = st.slider("Effective Tax Rate (%)", 30, 50, 36, 1)
    tax_rate     = tax_rate_pct / 100.0

# ── 6) Compute open vs closed positions ───────────────────────────────────
def signed_qty(row):
    if "Sell to Open" in row.ActionType:
        return +row.Quantity
    if "Buy to Close" in row.ActionType or "Expir" in row.ActionType:
        return -row.Quantity
    return 0

opts['SignedQty'] = opts.apply(signed_qty, axis=1)

grp = (
    opts
      .groupby(['Underlying','Expiry','Strike','OptType'])['SignedQty']
      .sum().reset_index()
)
open_positions = grp[grp['SignedQty']>0] \
                    .rename(columns={'SignedQty':'Quantity'}) \
                    [['Underlying','Expiry','Strike','OptType','Quantity']]

net_map = grp.set_index(['Underlying','Expiry','Strike','OptType'])['SignedQty']
opts['PositionStatus'] = opts.apply(
    lambda r: "Open"
      if net_map.loc[(r.Underlying,r.Expiry,r.Strike,r.OptType)]>0
      else "Expired/Closed",
    axis=1
)

# ── 7) Monthly pivot & chart data ─────────────────────────────────────────
pivot = (
    opts
      .groupby(['Month','OptType'])['Quantity']
      .sum().abs()
      .unstack(fill_value=0)
)
pivot['Total']        = pivot.sum(axis=1)
pivot['PreTaxProfit'] = opts.groupby('Month')['CashFlow'].sum()
pivot['ProfitPerCtr'] = pivot['PreTaxProfit']/pivot['Total']
pivot = pivot[['P','C','Total','PreTaxProfit','ProfitPerCtr']].reset_index()

chart_df               = pivot.copy()
chart_df['MonthLabel'] = pd.to_datetime(chart_df['Month'] + "-01") \
                             .dt.strftime("%B %Y")
month_order = chart_df['MonthLabel'].tolist()

# ── 8) Build Excel in-memory ──────────────────────────────────────────────
ytd         = opts['CashFlow'].sum()
days_traded = max((pd.to_datetime('today') - opts['Date'].min()).days, 1)
weekly_pre  = ytd / (days_traded / 7)
tax_e       = ytd * tax_rate
post_tax    = ytd - tax_e

summary_df = pd.DataFrame({
    "Metric": [
        "YTD Pre-Tax Gain",
        "Weekly Pre-Tax Gain",
        f"Tax Expense ({tax_rate_pct}% Effective Rate)",
        "YTD Post-Tax Gain"
    ],
    "Value": [
        ytd,
        weekly_pre,
        -tax_e,
        post_tax
    ]
})

trade_df = opts.copy()
trade_df['Expiry'] = trade_df['Expiry'].dt.strftime("%B %d, %Y").replace(r"\b0","", regex=True)
trade_df['Date']   = trade_df['Date'].dt.strftime("%B %d, %Y").replace(r"\b0","", regex=True)

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    summary_df.to_excel(writer, sheet_name="YTD Summary", index=False)
    open_positions.to_excel(writer, sheet_name="Current Exposure", index=False)
    trade_df[[ 'Underlying','Expiry','Date','ActionType',
               'Strike','OptType','Quantity','Price',
               'CashFlow','PositionStatus' ]].to_excel(
        writer, sheet_name="Trade Detail", index=False
    )
excel_bytes = excel_buffer.getvalue()

# ── 9) PDF export function ─────────────────────────────────────────────────
def create_pdf(opts_df, chart_df, month_order, tax_rate, tax_rate_pct):
    fig = plt.figure(figsize=(8,10), facecolor='white')
    today = datetime.date.today()
    suf = "th" if 11 <= today.day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(today.day % 10, "th")
    date_str = today.strftime(f"%B {today.day}{suf}, %Y")

    ax1 = fig.add_axes([0.1, 0.82, 0.8, 0.16])
    ax1.axis('off')
    ax1.text(0.5, 0.75,
             "Year-to-Date Options Income Summary",
             ha='center', va='center',
             fontsize=14,
             fontfamily='Times New Roman',
             fontweight='bold',
             color='black')
    ax1.text(0.5, 0.62,
             date_str,
             ha='center', va='center',
             fontsize=14,
             fontfamily='Times New Roman',
             color='black')

    stats = [
        ("YTD Gain Post-Fees:",      f"${ytd:,.1f}"),
        ("Weekly Pre-Tax:",          f"${weekly_pre:,.1f}"),
        (f"Tax Expense ({tax_rate_pct}%):", f"-${tax_e:,.1f}"),
        ("YTD Post-Tax Gain:",       f"${post_tax:,.1f}")
    ]

    for i, (lbl, val) in enumerate(stats):
        y = 0.44 - i * 0.10
        ax1.text(0.48, y,
                 lbl,
                 ha='right', va='center',
                 fontsize=14,
                 fontfamily='Times New Roman',
                 fontweight='bold',
                 color='black')
        ax1.text(0.52, y,
                 val,
                 ha='left', va='center',
                 fontsize=14,
                 fontfamily='Times New Roman',
                 color='black')

    ax2 = fig.add_axes([0.1, 0.25, 0.8, 0.45])
    df2 = chart_df.set_index('MonthLabel').reindex(month_order)

    fill_color    = (226/255, 239/255, 218/255)
    outline_color = (55/255,  86/255,  35/255)

    bars = ax2.bar(df2.index, df2['PreTaxProfit'],
                   color=fill_color,
                   edgecolor=outline_color,
                   linewidth=1.5)
    for bar in bars:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            h/2,
            f"${h:,.1f}",
            ha='center',
            va='center',
            color=outline_color,
            fontfamily='Arial',
            fontsize=12
        )

    ax2.set_ylim(0, df2['PreTaxProfit'].max() * 1.1)
    ax2.set_title("Monthly P&L Exhibit:",
                  fontfamily='Times New Roman',
                  fontsize=14,
                  color='black',
                  pad=12)
    ax2.set_ylabel("Pre-Tax Profit",
                   fontfamily='Arial',
                   fontsize=12,
                   color='black')
    ax2.yaxis.set_ticks([0])

    for spine in ['top','right']:
        ax2.spines[spine].set_visible(False)
    ax2.spines['left'].set_color('black')
    ax2.spines['bottom'].set_color('black')
    ax2.tick_params(axis='y', left=False, labelleft=False)
    ax2.tick_params(axis='x',
                    colors='black',
                    labelrotation=45,
                    labelsize=12)
    ax2.set_xticklabels(month_order,
                        ha='right',
                        fontfamily='Arial',
                        fontsize=12)

    fig.tight_layout(pad=2)

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(fig, facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()

pdf_bytes = create_pdf(opts, chart_df, month_order, tax_rate, tax_rate_pct)

# ── 10) Download buttons ──────────────────────────────────────────────────
c1, c2 = st.columns(2, gap="small")
with c1:
    st.download_button("📥 Export to Excel",
                       excel_bytes,
                       "options_pnl.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with c2:
    st.download_button("📥 Download PDF Summary",
                       pdf_bytes,
                       "options_summary.pdf",
                       "application/pdf")

st.markdown("---")

# ── 11) In-app YTD summary ─────────────────────────────────────────────────
st.subheader("Year-to-Date Summary")
col1, col2, col3 = st.columns(3)
col1.metric("YTD Gain Post-Fees",           f"${ytd:,.1f}")
col2.metric("Weekly Pre-Tax",               f"${weekly_pre:,.1f}")
col3.metric(f"Tax Expense ({tax_rate_pct}%)", f"-${tax_e:,.1f}")
st.metric("YTD Post-Tax Gain",              f"${post_tax:,.1f}")

st.markdown("---")

# ── 12) In-app Monthly P&L chart ────────────────────────────────────────────
st.subheader("Monthly P&L")
fill_hex    = "#E2EFDA"
outline_hex = "#375623"
bars = (
    alt.Chart(chart_df)
       .mark_bar(fill=fill_hex, stroke=outline_hex, strokeWidth=1.5)
       .encode(
           x=alt.X("MonthLabel:N", sort=month_order,
                   axis=alt.Axis(labelAngle=0,
                                 labelColor="white",
                                 domainColor="white",
                                 tickColor="white",
                                 title=None)),
           y=alt.Y("PreTaxProfit:Q",
                   axis=alt.Axis(title="Pre-Tax Profit",
                                 labels=False, grid=False))
       )
)
labels = (
    alt.Chart(chart_df)
       .transform_calculate(Mid="datum.PreTaxProfit/2")
       .mark_text(align="center", baseline="middle",
                  color=outline_hex, font="Arial")
       .encode(
           x=alt.X("MonthLabel:N", sort=month_order),
           y=alt.Y("Mid:Q"),
           text=alt.Text("PreTaxProfit:Q", format="$,.1f")
       )
)
st.altair_chart((bars + labels).configure_view(strokeWidth=0),
                use_container_width=True)

st.markdown("---")

# ── 13) In-app Current Exposure table ─────────────────────────────────────
st.subheader("Current Exposure")
exp_df = open_positions.copy()
exp_df['Expiry'] = exp_df['Expiry'].dt.strftime("%B %d, %Y").replace(r"\b0","", regex=True)
exp_df['Strike'] = exp_df['Strike'].map(lambda x: f"${x:,.1f}")
exp_df = exp_df.rename(columns={"OptType":"P/C"}).reset_index(drop=True)
st.dataframe(exp_df[["Underlying","Expiry","Strike","P/C","Quantity"]],
             use_container_width=True)

st.markdown("---")

# ── 14) In-app Transactions detail with color coding ──────────────────────
st.subheader("Transactions")
t = opts.copy()
t['Expiry']           = t['Expiry'].dt.strftime("%B %d, %Y").replace(r"\b0","", regex=True)
t['Transaction Date'] = t['Date'].dt.strftime("%B %d, %Y").replace(r"\b0","", regex=True)
t['Strike']           = t['Strike'].map(lambda x: f"${x:,.1f}")
t['Price']            = t['Price'].map(lambda x: f"${x:,.3f}")
def fmt_amt(x):
    return f"(${abs(x):,.1f})" if x<0 else f"${x:,.1f}"
t['Amount'] = t['CashFlow'].map(fmt_amt)

display = pd.DataFrame({
    "Underlying":       t['Underlying'],
    "Expiry":           t['Expiry'],
    "Transaction Date": t['Transaction Date'],
    "Action":           t['ActionType'],
    "Strike":           t['Strike'],
    "P/C":              t['OptType'],
    "Quantity":         t['Quantity'],
    "Price":            t['Price'],
    "Amount":           t['Amount'],
    "Position Status":  t['PositionStatus']
})

# Define row-wise styling
def highlight_row(row):
    if row["Action"] == "Sell to Open":
        return ["color: #008000"] * len(row)    # green
    elif row["Action"] == "Buy to Close":
        return ["color: #FF0000"] * len(row)    # red
    elif "Expired" in row["Position Status"]:
        return ["color: #FF8C00"] * len(row)    # dark orange
    else:
        return [""] * len(row)

styled = display.reset_index(drop=True).style.apply(highlight_row, axis=1)

st.dataframe(styled, use_container_width=True)