import io
import datetime
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#. .\.venv\Scripts\Activate.ps1 <<<Paste into Windows Powershell
# ── 1) Page setup & global font ──────────────────────────────────────────
st.set_page_config(page_title="Options Income P&L Summary", layout="wide")
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

      .welcome-title {
        font-family: "Times New Roman", Times, serif !important;
        font-weight: 200;
        font-size: 3.8rem;
        letter-spacing: -0.5px;
        line-height: 1.05;
        text-align: center;
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
        <div class="welcome-title">
            Welcome to<br>Options Income P&L
        </div>
    </div>
    """,
    unsafe_allow_html=True
    )

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded = st.file_uploader(
        label="Upload CSV",
        type="csv",
        label_visibility="hidden"  # only one file allowed by default
    )

if not uploaded:
    # small creator credit at very bottom of landing page
    st.markdown(
        '<p style="text-align: center; color: #888888; font-size: 12px; margin-top: 10px;">'
        'Created by Justin Zhao'
        '</p>',
        unsafe_allow_html=True
    )
    st.stop()

# ── 3) Main title once files are in ────────────────────────────────────────
st.title("Options P&L Summary")

# ── 4) CSV parsing & cleaning ──────────────────────────────────────────────
def process(df):
    opts = df[df['Description'].str.startswith(('PUT','CALL'))].copy()
    opts = opts[opts['Action'].isin(['Sell to Open','Buy to Close','Expired'])].copy()
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

# read single uploaded CSV
df   = pd.read_csv(uploaded)
opts = process(df)

# ── 5) Settings expander ─────────────────────────────────────────────
with st.expander("Tax Rate Setting", expanded=False):
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

# aggregate net signed quantity per contract
grp = (
    opts
      .groupby(['Underlying','Expiry','Strike','OptType'])['SignedQty']
      .sum()
      .reset_index()
)

# build current exposures (net > 0)
open_positions = (
    grp[grp['SignedQty'] > 0]
      .rename(columns={'SignedQty':'Quantity'})
      [['Underlying','Expiry','Strike','OptType','Quantity']]
)

# ── 6b) expiry‐override: drop any contract whose Expiry is before today
today = pd.to_datetime('today').normalize()
open_positions = open_positions[open_positions['Expiry'] >= today]

# build a lookup map for net qty
net_map = grp.set_index(['Underlying','Expiry','Strike','OptType'])['SignedQty']

# assign PositionStatus on every trade, but force Expired/Closed if past expiry
def get_status(r):
    if r.Expiry < today:
        return "Expired/Closed"
    return "Open" if net_map.loc[(r.Underlying, r.Expiry, r.Strike, r.OptType)] > 0 \
                  else "Expired/Closed"

opts['PositionStatus'] = opts.apply(get_status, axis=1)

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

# ── Limit Monthly P&L to last 13 months maximum ─────────────────────────
pivot['Period'] = pd.to_datetime(pivot['Month'] + "-01").dt.to_period('M')
latest_period = pivot['Period'].max()
earliest_period = latest_period - 12
pivot = pivot[(pivot['Period'] >= earliest_period) & (pivot['Period'] <= latest_period)].copy()
pivot.drop(columns=['Period'], inplace=True)

chart_df = pivot.copy()
chart_df['MonthLabel'] = pd.to_datetime(chart_df['Month'] + "-01") \
                             .dt.strftime("%B %Y")
chart_df['IsCurrent'] = chart_df['Month'] == datetime.date.today().strftime("%Y-%m")
month_order = chart_df['MonthLabel'].tolist()

# ——— Compute your summary_df and trade_df just once ———
current_year = datetime.date.today().year
opts_current_year = opts[opts['Date'].dt.year == current_year]
ytd         = opts_current_year['CashFlow'].sum()
days_traded = max((pd.to_datetime('today') - opts['Date'].min()).days, 1)
weekly_pre  = ytd / (days_traded / 7)
tax_e       = ytd * tax_rate
post_tax    = ytd - tax_e

current_month_str = datetime.date.today().strftime("%Y-%m")
completed_months = pivot[
    (pivot['Month'].str.startswith(str(current_year))) &
    (pivot['Month'] < current_month_str)
]
avg_monthly_pre_tax = (
    completed_months['PreTaxProfit'].mean()
    if not completed_months.empty else 0
)

# This is your YTD summary (numeric Value)
summary_df = pd.DataFrame({
    "Metric": [
        "YTD Pre-Tax Gain",
        "Weekly Pre-Tax Gain",
        f"Tax Expense ({tax_rate_pct}% Effective Rate)",
        "YTD Post-Tax Gain"
    ],
    "Value": [ytd, weekly_pre, -tax_e, post_tax]
})

# Copy opts into trade_df but KEEP Date/Expiry as real datetimes
trade_df = opts.copy()

# ── Now write all three sheets with proper formatting ───────────────────────────
excel_buffer = io.BytesIO()
with pd.ExcelWriter(
    excel_buffer,
    engine="xlsxwriter",
    date_format='yyyy-mm-dd',
    datetime_format='yyyy-mm-dd'
) as writer:
    # Sheet 1: YTD Summary
    summary_df.to_excel(writer, sheet_name="YTD Summary", index=False)

    # Sheet 2: Current Exposure (Expiry is datetime, Quantity numeric)
    open_positions.to_excel(writer, sheet_name="Current Exposure", index=False)

    # Sheet 3: Trade Detail
    trade_df[
       ['Underlying','Expiry','Date','ActionType',
        'Strike','OptType','Quantity','Price','CashFlow','PositionStatus']
    ].to_excel(writer, sheet_name="Trade Detail", index=False)

    # Grab the workbook & define formats
    workbook  = writer.book
    date_fmt  = workbook.add_format({ 'num_format': 'yyyy-mm-dd' })
    money_fmt = workbook.add_format({ 'num_format': '$#,##0.00' })
    int_fmt   = workbook.add_format({ 'num_format': '0' })

    # — Format YTD Summary —
    ws1 = writer.sheets['YTD Summary']
    ws1.set_column('B:B', 18, money_fmt)     # Value column

    # — Format Current Exposure —
    ws2 = writer.sheets['Current Exposure']
    ws2.set_column('B:B', 15, date_fmt)      # Expiry
    ws2.set_column('E:E', 10, int_fmt)       # Quantity

    # — Format Trade Detail —
    ws3 = writer.sheets['Trade Detail']
    ws3.set_column('B:C', 15, date_fmt)      # Expiry (B), Date (C)
    ws3.set_column('E:E', 10, int_fmt)       # Strike
    ws3.set_column('G:G', 10, int_fmt)       # Quantity
    ws3.set_column('H:I', 12, money_fmt)     # Price (H), CashFlow (I)

# Pull out the bytes for the download button
excel_bytes = excel_buffer.getvalue()

# ── 9) PDF export function (two tables + chart + footer metrics) ─────────────────
def create_pdf(opts_df, chart_df, month_order, tax_rate, tax_rate_pct):
    import io
    import datetime
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
    today = datetime.date.today()

    suf = "th" if 11 <= today.day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(today.day % 10, "th")
    date_str = today.strftime(f"%B {today.day}{suf}, %Y")

    # --- compute the two table datasets ---
    left_data = [
        ["YTD Pre-Tax Gain (Includes Fees)",   f"${ytd:,.1f}"],
        [f"Tax Expense ({tax_rate_pct}%)",     f"(${tax_e:,.1f})"],
        ["YTD Post-Tax Gain",                  f"${post_tax:,.1f}"],
        ["Average Monthly Pre-Tax Gain",       f"${avg_monthly_pre_tax:,.1f}"]
    ]

    # Period metrics (only completed months)
    df2 = chart_df.set_index('MonthLabel').reindex(month_order)
    completed = df2.loc[~df2['IsCurrent'], 'PreTaxProfit']
    total_period = completed.sum()
    avg_period   = completed.mean() if not completed.empty else 0
    completed_labels = [lbl for lbl, is_cur in zip(month_order, df2['IsCurrent']) if not is_cur]
    period_str = f"{completed_labels[0]} – {completed_labels[-1]}" if completed_labels else ""

    right_data = [
        ["Total Pre-Tax Gain in Period",        f"${total_period:,.1f}"],
        ["Avg. Monthly Pre-Tax Gain in Period", f"${avg_period:,.1f}"]
    ]

    axL = fig.add_axes([0.05, 0.78, 0.45, 0.17])
    axL.axis('off')

    tblL = axL.table(
        cellText=[
            ["Year-to-Date Income Summary", ""],
            [f"({date_str})", ""],
            *left_data
        ],
        colWidths=[0.65, 0.35],
        loc='upper left'
    )
    tblL.auto_set_font_size(False)
    tblL.set_fontsize(10)
    tblL.scale(1, 1.5)

    for cell in tblL.get_celld().values():
        cell.set_edgecolor('black')
        cell.set_linewidth(1)

    for (r, c), cell in tblL.get_celld().items():
        if r == 0:   # title row
            cell.set_text_props(ha='center', va='center', fontfamily='Times New Roman', fontweight='bold')
        elif r == 1: # date row
            cell.set_text_props(ha='center', va='center', fontfamily='Times New Roman', fontweight='normal')
        else:        # data rows
            cell.set_text_props(ha=('left' if c == 0 else 'center'), fontfamily='Times New Roman', va='center')

    # --- right table (Monthly Pre-Tax P&L) ---
    axR = fig.add_axes([0.52, 0.78, 0.43, 0.17])
    axR.axis('off')
    tblR = axR.table(
        cellText=[
            ["Monthly Pre-Tax P&L", ""],
            [f"({period_str})", ""],
            *right_data
        ],
        colWidths=[0.65, 0.35],
        loc='upper left'
    )
    tblR.auto_set_font_size(False)
    tblR.set_fontsize(10)
    tblR.scale(1, 1.5)
    for cell in tblR.get_celld().values():
        cell.set_edgecolor('black')
        cell.set_linewidth(1)
    for (r, c), cell in tblR.get_celld().items():
        if r == 0:
            cell.set_text_props(ha='center', va='center', fontfamily='Times New Roman', fontweight='bold')
        elif r == 1:
            cell.set_text_props(ha='center', va='center', fontfamily='Times New Roman', fontweight='normal')
        else:
            cell.set_text_props(ha=('left' if c == 0 else 'center'), va='center', fontfamily='Times New Roman',)

    ax2 = fig.add_axes([0.1, 0.10, 0.8, 0.60])

    fill_color    = (226/255, 239/255, 218/255)
    outline_color = (55/255,  86/255,  35/255)
    current_fill  = "#FFFFCC"
    current_edge  = "#FF9900"
    current_text  = "#FF9900"

    bars = ax2.bar(
        df2.index,
        df2['PreTaxProfit'],
        color=[current_fill if is_cur else fill_color for is_cur in df2['IsCurrent']],
        edgecolor=[current_edge if is_cur else outline_color for is_cur in df2['IsCurrent']],
        linewidth=1.5
    )
    for bar, is_cur in zip(bars, df2['IsCurrent']):
        h = bar.get_height()
        label = f"${h/1000:,.1f}K"
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            h/2,
            label,
            ha='center', va='center',
            color=(current_text if is_cur else outline_color),
            fontfamily='Times New Roman',
            fontsize=10
        )

    ax2.set_ylim(0, df2['PreTaxProfit'].max() * 1.1)
    ax2.set_title(
        "Monthly Pre-Tax P&L Exhibit:",
        fontfamily='Times New Roman', fontsize=12,
        color='black', pad=12, fontweight='bold'
    )

    ax2.yaxis.set_visible(False)
    for spine in ['left','top','right']:
        ax2.spines[spine].set_visible(False)
    ax2.spines['bottom'].set_color('black')
    ax2.set_xticks(range(len(month_order)))
    ax2.set_xticklabels(
        month_order, rotation=90, ha='center',
        fontfamily='Times New Roman', fontsize=10
    )

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(fig, facecolor='white', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()

# ── 9b) Generate PDF bytes ─────────────────────────────────────────────────
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
st.subheader(f"{current_year} Year-to-Date Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("YTD Pre-Tax Gain",            f"${ytd:,.1f}")
col2.metric(f"Tax Expense ({tax_rate_pct}%)", f"-${tax_e:,.1f}")
col3.metric("YTD Post-Tax Gain",           f"${post_tax:,.1f}")
col4.metric("Average Monthly Pre-Tax Gain", f"${avg_monthly_pre_tax:,.1f}")

st.markdown("---")

# ── 12) In-app Monthly P&L chart ────────────────────────────────────────────
st.subheader("Monthly Pre-Tax P&L")
fill_hex    = "#E2EFDA"
outline_hex = "#375623"
current_fill  = "#FFFFCC"
current_edge  = "#FF9900"
current_text  = "#FF9900"

chart_df['IsCurrent'] = chart_df['Month'] == datetime.date.today().strftime("%Y-%m")
chart_avg = chart_df['PreTaxProfit'].iloc[:-1].mean() if len(chart_df) > 1 else 0

bars = (
    alt.Chart(chart_df)
       .mark_bar(strokeWidth=1.5)
       .encode(
           x=alt.X(
               "MonthLabel:N",
               sort=month_order,
               axis=alt.Axis(
                    labelAngle=0,
                    labelColor="white",
                    labelFontSize=16,
                    labelLineHeight=16,
                    labelLimit=0,
                    labelFontWeight="bold",
                    domain=False,
                    tickColor="white",
                    title=None,
                    labelExpr="split(datum.value, ' ')"
               )
           ),
           y=alt.Y(
               "PreTaxProfit:Q",
               axis=alt.Axis(domain=False, ticks=False, labels=False, title=None)
           ),
           fill=alt.condition(
               alt.datum.IsCurrent,
               alt.value(current_fill),
               alt.value(fill_hex)
           ),
           stroke=alt.condition(
               alt.datum.IsCurrent,
               alt.value(current_edge),
               alt.value(outline_hex)
           )
       )
)

labels = (
    alt.Chart(chart_df)
       .transform_calculate(
           mid="datum.PreTaxProfit / 2",
           k=" '$' + format(datum.PreTaxProfit/1000, '.1f') + 'K'"
       )
       .mark_text(align="center", baseline="middle", fontSize=16, fontWeight="normal")
       .encode(
           x=alt.X("MonthLabel:N", sort=month_order),
           y=alt.Y("mid:Q"),
           text=alt.Text("k:N"),
           color=alt.condition(
               alt.datum.IsCurrent,
               alt.value(current_text),
               alt.value(outline_hex)
           )
       )
)

avg_line = (
    alt.Chart(pd.DataFrame({'avg': [chart_avg]}))
       .mark_rule(color="#008000", size=1, strokeDash=[5,5])
       .encode(y="avg:Q")
)

label_bg = (
    alt.Chart(pd.DataFrame({
        "MonthLabel": [month_order[-1]],
        "avg": [chart_avg]
    }))
    .mark_rect(width=120, height=25, fill="black", stroke="white", strokeWidth=1)
    .encode(
        x=alt.X("MonthLabel:N", sort=month_order),
        y=alt.Y("avg:Q")
    )
)

avg_label = (
    alt.Chart(pd.DataFrame({
        "MonthLabel": [month_order[-1]],
        "avg": [chart_avg],
        "label": ["Monthly Avg."]
    }))
    .mark_text(align="center", baseline="middle",
               dx=0, dy=-1,
               font="Arial", fontWeight="normal", fontStyle="italic",
               fontSize=16, color="white")
    .encode(
        x=alt.X("MonthLabel:N", sort=month_order),
        y=alt.Y("avg:Q"),
        text=alt.Text("label:N")
    )
)

st.altair_chart(
    (bars + labels + avg_line + label_bg + avg_label)
      .configure_view(strokeWidth=0)
      .configure_axis(labelColor="white"),
    use_container_width=True
)

# ── 13) Period metrics below the chart ────────────────────────────────────
current_month_str = datetime.date.today().strftime("%Y-%m")
if chart_df['Month'].iloc[-1] == current_month_str:
    completed = chart_df.iloc[:-1]
else:
    completed = chart_df.copy()

start_label = completed['MonthLabel'].iloc[0]
end_label   = completed['MonthLabel'].iloc[-1]
date_range  = f"{start_label} - {end_label}"

total_period = completed['PreTaxProfit'].sum()
avg_period   = completed['PreTaxProfit'].mean() if len(completed) > 0 else 0

col_avg, col_total = st.columns(2)
with col_avg:
    st.metric(
        f"Average Monthly Pre-Tax Gain in Period ({date_range})",
        f"${avg_period:,.1f}"
    )
with col_total:
    st.metric(
        f"Total Pre-Tax Gain in Period ({date_range})",
        f"${total_period:,.1f}"
    )

st.markdown("---")

# ── 14) In-app Current Exposure table ──────────────────────────────────────
today_str = datetime.date.today().strftime("%B %d, %Y").replace(" 0", " ")
st.subheader(f"Current Exposure as of {today_str}")
exp_df = open_positions.copy()
exp_df['Expiry'] = exp_df['Expiry'].dt.strftime("%B %d, %Y").replace(" 0", " ")
exp_df['Strike'] = exp_df['Strike'].map(lambda x: f"${x:,.1f}")
exp_df = exp_df.rename(columns={"OptType":"P/C"}).reset_index(drop=True)
st.dataframe(
    exp_df[["Underlying","Expiry","Strike","P/C","Quantity"]],
    use_container_width=True
)

st.markdown("---")

# ── 15) In-app Transactions detail ─────────────────────────────────────────
st.subheader("Transactions")
t = opts.copy()
t['Expiry']           = t['Expiry'].dt.strftime("%B %d, %Y").replace(" 0", " ")
t['Transaction Date'] = t['Date'].dt.strftime("%B %d, %Y").replace(" 0", " ")
t['Strike']           = t['Strike'].map(lambda x: f"${x:,.1f}")
t['Price']            = t['Price'].map(lambda x: f"${x:,.3f}")

def fmt_amt(x):
    return f"(${abs(x):,.1f})" if x < 0 else f"${x:,.1f}"
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

def highlight_row(row):
    if row["Action"] == "Sell to Open":
        return ["background-color: #233723"] * len(row)
    if row["Action"] == "Buy to Close":
        return ["background-color: #431B1D"] * len(row)
    return ["background-color: #4E3B1E"] * len(row)

styled = display.reset_index(drop=True).style.apply(highlight_row, axis=1)
st.dataframe(styled, use_container_width=True)