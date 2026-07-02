# -*- coding: utf-8 -*-
import csv
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
FIELDS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in FIELDS:
            if key not in ("ts_code", "trade_date"):
                row[key] = float(row[key])
    rows.sort(key=lambda r: r["trade_date"])
    return rows


def mean(values):
    return sum(values) / len(values) if values else 0


def median(values):
    if not values:
        return 0
    vals = sorted(values)
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


def enrich_rows(rows):
    peak = rows[0]["close"]
    for i, row in enumerate(rows):
        closes = [r["close"] for r in rows[max(0, i - 4) : i + 1]]
        closes20 = [r["close"] for r in rows[max(0, i - 19) : i + 1]]
        returns20 = [r["pct_chg"] for r in rows[max(0, i - 19) : i + 1]]
        peak = max(peak, row["close"])
        row["ma5"] = mean(closes)
        row["ma20"] = mean(closes20)
        row["vol_ma20"] = mean([r["vol"] for r in rows[max(0, i - 19) : i + 1]])
        row["drawdown_pct"] = (row["close"] / peak - 1) * 100
        row["volatility20"] = (
            (mean([(x - mean(returns20)) ** 2 for x in returns20]) ** 0.5)
            if len(returns20) > 1
            else 0
        )
    return rows


def monthly_returns(rows):
    months = {}
    for row in rows:
        key = row["trade_date"][:6]
        months.setdefault(key, []).append(row)
    out = []
    for key in sorted(months):
        mrows = months[key]
        pct = (mrows[-1]["close"] / mrows[0]["close"] - 1) * 100
        out.append({"month": f"{key[:4]}-{key[4:]}", "return_pct": pct})
    return out


def summary_for(name, code, csv_name, rows):
    returns = [row["pct_chg"] for row in rows]
    gain_days = len([x for x in returns if x > 0])
    loss_days = len([x for x in returns if x < 0])
    flat_days = len(rows) - gain_days - loss_days
    max_dd = min(row["drawdown_pct"] for row in rows)
    best = max(rows, key=lambda r: r["pct_chg"])
    worst = min(rows, key=lambda r: r["pct_chg"])
    highest_vol = max(rows, key=lambda r: r["vol"])
    pct = (rows[-1]["close"] / rows[0]["close"] - 1) * 100
    return {
        "market": name,
        "code": code,
        "csv": csv_name,
        "rows": len(rows),
        "start": rows[0]["trade_date"],
        "end": rows[-1]["trade_date"],
        "first_close": rows[0]["close"],
        "last_close": rows[-1]["close"],
        "close_change_pct": pct,
        "avg_return_pct": mean(returns),
        "median_return_pct": median(returns),
        "max_drawdown_pct": max_dd,
        "up_days": gain_days,
        "down_days": loss_days,
        "flat_days": flat_days,
        "up_ratio_pct": gain_days / len(rows) * 100,
        "avg_vol": mean([row["vol"] for row in rows]),
        "avg_amount": mean([row["amount"] for row in rows]),
        "total_vol": sum(row["vol"] for row in rows),
        "total_amount": sum(row["amount"] for row in rows),
        "best_day": {
            "date": best["trade_date"],
            "pct_chg": best["pct_chg"],
            "close": best["close"],
        },
        "worst_day": {
            "date": worst["trade_date"],
            "pct_chg": worst["pct_chg"],
            "close": worst["close"],
        },
        "highest_vol_day": {
            "date": highest_vol["trade_date"],
            "vol": highest_vol["vol"],
            "close": highest_vol["close"],
        },
    }


def build_html(data, summaries):
    template = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>中芯国际 A/H 股近一年交易数据分析</title>
<style>
:root{--bg:#f5f6f8;--panel:#fff;--ink:#182230;--muted:#667085;--border:#d0d5dd;--grid:#eaecf0;--red:#d92d20;--green:#039855;--blue:#2563eb;--violet:#7c3aed;--amber:#d97706}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}.wrap{max-width:1360px;margin:0 auto;padding:24px}.top{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:18px}h1{font-size:26px;margin:0 0 6px}.sub,.note{color:var(--muted)}.tabs{display:flex;gap:8px;flex-wrap:wrap}.tab{border:1px solid var(--border);background:#fff;color:var(--ink);padding:8px 14px;border-radius:8px;cursor:pointer}.tab.active{background:#111827;color:#fff;border-color:#111827}.cards{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;margin-bottom:16px}.metric{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px}.metric .k{color:var(--muted);font-size:12px}.metric .v{font-size:18px;font-weight:750;margin-top:4px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:14px}.wide{grid-column:1/-1}.panel-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:8px}.title{font-weight:750;font-size:16px}.caption{color:var(--muted);font-size:12px;margin-top:2px}.interpret{border-left:3px solid #98a2b3;background:#f9fafb;color:#344054;padding:8px 10px;margin-top:10px;min-height:40px}.chart{width:100%;height:320px;display:block}.small{height:260px}.table{width:100%;border-collapse:collapse}.table th,.table td{border-bottom:1px solid var(--grid);padding:8px;text-align:left}.table th{color:var(--muted);font-weight:650}.downloads{display:flex;gap:10px;align-items:center}.downloads a{color:#175cd3;text-decoration:none}.downloads a:hover{text-decoration:underline}@media(max-width:980px){.grid{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.top{display:block}.tabs{margin-top:12px}.wrap{padding:16px}.chart{height:280px}}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div>
      <h1>中芯国际 A/H 股近一年交易数据分析</h1>
      <div class="sub">基于本地 CSV：2025-07-02 至 2026-07-02。A 股与港股独立分析，并提供归一化收盘价对比。</div>
    </div>
    <div class="tabs">
      <button class="tab active" data-market="A股">A 股 688981.SH</button>
      <button class="tab" data-market="港股">港股 00981.HK</button>
    </div>
  </div>
  <div class="cards" id="metrics"></div>
  <div class="grid">
    <section class="panel wide"><div class="panel-head"><div><div class="title">图 1 K 线与 5/20 日均线</div><div class="caption">观察价格区间、趋势方向和短中期均线关系</div></div></div><canvas id="kline" class="chart"></canvas><div class="interpret" id="i-kline"></div></section>
    <section class="panel"><div class="panel-head"><div><div class="title">图 2 成交量与 20 日均量</div><div class="caption">识别放量交易日和量能变化</div></div></div><canvas id="volume" class="chart small"></canvas><div class="interpret" id="i-volume"></div></section>
    <section class="panel"><div class="panel-head"><div><div class="title">图 3 收盘价与回撤</div><div class="caption">蓝线为收盘价，紫线为相对阶段高点的回撤</div></div></div><canvas id="closeDrawdown" class="chart small"></canvas><div class="interpret" id="i-drawdown"></div></section>
    <section class="panel"><div class="panel-head"><div><div class="title">图 4 日收益率分布</div><div class="caption">统计每日涨跌幅落点，观察尾部风险</div></div></div><canvas id="returnHist" class="chart small"></canvas><div class="interpret" id="i-hist"></div></section>
    <section class="panel"><div class="panel-head"><div><div class="title">图 5 20 日滚动波动率</div><div class="caption">用近 20 个交易日涨跌幅标准差衡量短期波动</div></div></div><canvas id="volatility" class="chart small"></canvas><div class="interpret" id="i-volatility"></div></section>
    <section class="panel"><div class="panel-head"><div><div class="title">图 6 成交额趋势</div><div class="caption">展示市场资金交易规模的变化</div></div></div><canvas id="amount" class="chart small"></canvas><div class="interpret" id="i-amount"></div></section>
    <section class="panel"><div class="panel-head"><div><div class="title">图 7 价量关系散点</div><div class="caption">横轴为成交量，纵轴为日涨跌幅</div></div></div><canvas id="scatter" class="chart small"></canvas><div class="interpret" id="i-scatter"></div></section>
    <section class="panel"><div class="panel-head"><div><div class="title">图 8 月度收益率</div><div class="caption">按自然月统计首尾收盘价涨跌</div></div></div><canvas id="monthly" class="chart small"></canvas><div class="interpret" id="i-monthly"></div></section>
    <section class="panel wide"><div class="panel-head"><div><div class="title">图 9 A 股与港股归一化收盘价对比</div><div class="caption">将各自首日收盘价设为 100，比较相对走势</div></div></div><canvas id="compare" class="chart"></canvas><div class="interpret" id="i-compare"></div></section>
    <section class="panel"><div class="panel-head"><div><div class="title">表 1 关键交易日</div><div class="caption">最大涨幅、最大跌幅、最高成交量</div></div></div><div id="keyTable"></div><div class="interpret" id="i-table"></div></section>
    <section class="panel"><div class="panel-head"><div><div class="title">数据文件</div><div class="caption">当前市场 CSV 与摘要文件</div></div></div><div class="downloads"><a id="csvLink" href="#">下载当前市场 CSV</a><span id="csvName" class="sub"></span></div><p class="note">统计图均由页面内嵌的本地 CSV 数据生成，无需联网。</p></section>
  </div>
</div>
<script>
const DATA = __DATA__;
const SUMMARIES = __SUMMARIES__;
let current = "A股";
const fmtDate=d=>d.slice(0,4)+"-"+d.slice(4,6)+"-"+d.slice(6);
const fmtNum=n=>Number(n).toLocaleString("zh-CN",{maximumFractionDigits:2});
const pct=n=>(Number(n)>=0?"+":"")+Number(n).toFixed(2)+"%";
function canvas(c){const r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;c.width=Math.max(300,Math.floor(r.width*d));c.height=Math.max(180,Math.floor(r.height*d));const ctx=c.getContext("2d");ctx.setTransform(d,0,0,d,0,0);return{ctx,w:r.width,h:r.height};}
function range(vals,p=.08){let lo=Math.min(...vals),hi=Math.max(...vals);if(lo===hi){lo-=1;hi+=1}let pad=(hi-lo)*p;return[lo-pad,hi+pad];}
function axes(ctx,w,h,m,y0,y1,labels){ctx.clearRect(0,0,w,h);ctx.strokeStyle="#eaecf0";ctx.fillStyle="#667085";ctx.lineWidth=1;ctx.font="12px Arial";for(let i=0;i<=4;i++){let y=m.t+(h-m.t-m.b)*i/4;ctx.beginPath();ctx.moveTo(m.l,y);ctx.lineTo(w-m.r,y);ctx.stroke();ctx.fillText(fmtNum(y1-(y1-y0)*i/4),8,y+4)}[0,Math.floor((labels.length-1)/2),labels.length-1].forEach(i=>{let x=m.l+(w-m.l-m.r)*(i/(labels.length-1||1));ctx.fillText(labels[i]||"",Math.max(4,Math.min(x-30,w-80)),h-8)});ctx.strokeStyle="#98a2b3";ctx.beginPath();ctx.moveTo(m.l,m.t);ctx.lineTo(m.l,h-m.b);ctx.lineTo(w-m.r,h-m.b);ctx.stroke();}
function line(ctx,rows,key,y0,y1,m,w,h,color,width=2){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();rows.forEach((r,i)=>{let x=m.l+(w-m.l-m.r)*(i/(rows.length-1||1));let y=m.t+(y1-r[key])/(y1-y0)*(h-m.t-m.b);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();}
function drawK(rows){let {ctx,w,h}=canvas(document.getElementById("kline")),m={l:64,r:18,t:16,b:34};let labels=rows.map(r=>fmtDate(r.trade_date)),[lo,hi]=range(rows.flatMap(r=>[r.high,r.low,r.ma5,r.ma20]));axes(ctx,w,h,m,lo,hi,labels);let cw=(w-m.l-m.r)/rows.length;rows.forEach((r,i)=>{let x=m.l+cw*i+cw/2,y=v=>m.t+(hi-v)/(hi-lo)*(h-m.t-m.b),up=r.close>=r.open;ctx.strokeStyle=up?"#d92d20":"#039855";ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.moveTo(x,y(r.high));ctx.lineTo(x,y(r.low));ctx.stroke();ctx.fillRect(x-Math.max(1,cw*.3),Math.min(y(r.open),y(r.close)),Math.max(2,cw*.6),Math.max(1,Math.abs(y(r.open)-y(r.close))));});line(ctx,rows,"ma5",lo,hi,m,w,h,"#2563eb",1.5);line(ctx,rows,"ma20",lo,hi,m,w,h,"#d97706",1.5);}
function drawVolume(rows){let {ctx,w,h}=canvas(document.getElementById("volume")),m={l:64,r:18,t:16,b:34};let labels=rows.map(r=>fmtDate(r.trade_date)),hi=Math.max(...rows.map(r=>r.vol))*1.08;axes(ctx,w,h,m,0,hi,labels);let cw=(w-m.l-m.r)/rows.length;rows.forEach((r,i)=>{let x=m.l+cw*i+cw*.15,y=m.t+(hi-r.vol)/hi*(h-m.t-m.b);ctx.fillStyle=r.close>=r.open?"#d92d20":"#039855";ctx.fillRect(x,y,Math.max(1,cw*.7),h-m.b-y);});line(ctx,rows,"vol_ma20",0,hi,m,w,h,"#2563eb",1.5);}
function drawCloseDrawdown(rows){let {ctx,w,h}=canvas(document.getElementById("closeDrawdown")),m={l:64,r:18,t:16,b:34};let labels=rows.map(r=>fmtDate(r.trade_date)),[lo,hi]=range(rows.map(r=>r.close));axes(ctx,w,h,m,lo,hi,labels);line(ctx,rows,"close",lo,hi,m,w,h,"#2563eb",2);let ddMin=Math.min(...rows.map(r=>r.drawdown_pct));ctx.strokeStyle="#7c3aed";ctx.lineWidth=1.5;ctx.beginPath();rows.forEach((r,i)=>{let x=m.l+(w-m.l-m.r)*(i/(rows.length-1||1));let y=(h-m.b)-((r.drawdown_pct/ddMin)*(h-m.t-m.b));i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();}
function drawHist(rows){let {ctx,w,h}=canvas(document.getElementById("returnHist")),m={l:48,r:18,t:16,b:34};let vals=rows.map(r=>r.pct_chg),min=Math.floor(Math.min(...vals)/2)*2,max=Math.ceil(Math.max(...vals)/2)*2,bins=[];for(let a=min;a<max;a+=2)bins.push({label:`${a}~${a+2}`,lo:a,hi:a+2,count:0});vals.forEach(v=>{let b=bins.find(x=>v>=x.lo&&v<x.hi)||bins[bins.length-1];b.count++;});let hi=Math.max(...bins.map(b=>b.count))*1.15;axes(ctx,w,h,m,0,hi,bins.map(b=>b.label));let bw=(w-m.l-m.r)/bins.length;bins.forEach((b,i)=>{let y=m.t+(hi-b.count)/hi*(h-m.t-m.b);ctx.fillStyle=b.hi<=0?"#039855":b.lo>=0?"#d92d20":"#98a2b3";ctx.fillRect(m.l+bw*i+bw*.12,y,Math.max(2,bw*.76),h-m.b-y);});}
function drawSimpleLine(id,rows,key,color){let {ctx,w,h}=canvas(document.getElementById(id)),m={l:64,r:18,t:16,b:34};let labels=rows.map(r=>fmtDate(r.trade_date)),[lo,hi]=range(rows.map(r=>r[key]));if(key==="volatility20")lo=0;axes(ctx,w,h,m,lo,hi,labels);line(ctx,rows,key,lo,hi,m,w,h,color,2);}
function drawScatter(rows){let {ctx,w,h}=canvas(document.getElementById("scatter")),m={l:64,r:18,t:16,b:34};let [x0,x1]=range(rows.map(r=>r.vol)),[y0,y1]=range(rows.map(r=>r.pct_chg));axes(ctx,w,h,m,y0,y1,["低量","中位","高量"]);rows.forEach(r=>{let x=m.l+(r.vol-x0)/(x1-x0)*(w-m.l-m.r),y=m.t+(y1-r.pct_chg)/(y1-y0)*(h-m.t-m.b);ctx.fillStyle=r.pct_chg>=0?"rgba(217,45,32,.55)":"rgba(3,152,85,.55)";ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill();});}
function drawMonthly(obj){let {ctx,w,h}=canvas(document.getElementById("monthly")),m={l:56,r:18,t:16,b:34};let rows=obj.monthly,[lo,hi]=range(rows.map(r=>r.return_pct));lo=Math.min(lo,0);hi=Math.max(hi,0);axes(ctx,w,h,m,lo,hi,rows.map(r=>r.month.slice(5)));let bw=(w-m.l-m.r)/rows.length,zero=m.t+(hi-0)/(hi-lo)*(h-m.t-m.b);rows.forEach((r,i)=>{let y=m.t+(hi-r.return_pct)/(hi-lo)*(h-m.t-m.b);ctx.fillStyle=r.return_pct>=0?"#d92d20":"#039855";ctx.fillRect(m.l+bw*i+bw*.15,Math.min(y,zero),Math.max(2,bw*.7),Math.max(1,Math.abs(zero-y)));});}
function drawCompare(){let {ctx,w,h}=canvas(document.getElementById("compare")),m={l:64,r:18,t:16,b:34};let a=DATA["A股"].rows,hk=DATA["港股"].rows;let toIndex=rows=>rows.map(r=>({date:r.trade_date,value:r.close/rows[0].close*100}));let ai=toIndex(a),hi=toIndex(hk),vals=[...ai,...hi].map(r=>r.value),[lo,top]=range(vals);axes(ctx,w,h,m,lo,top,[fmtDate(a[0].trade_date),"",fmtDate(a[a.length-1].trade_date)]);function draw(arr,color){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();arr.forEach((r,i)=>{let x=m.l+(w-m.l-m.r)*(i/(arr.length-1||1)),y=m.t+(top-r.value)/(top-lo)*(h-m.t-m.b);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();}draw(ai,"#2563eb");draw(hi,"#7c3aed");ctx.fillStyle="#2563eb";ctx.fillText("A股",w-90,22);ctx.fillStyle="#7c3aed";ctx.fillText("港股",w-50,22);}
function corr(xs,ys){let n=Math.min(xs.length,ys.length),mx=xs.slice(0,n).reduce((a,b)=>a+b,0)/n,my=ys.slice(0,n).reduce((a,b)=>a+b,0)/n;let num=0,dx=0,dy=0;for(let i=0;i<n;i++){num+=(xs[i]-mx)*(ys[i]-my);dx+=(xs[i]-mx)**2;dy+=(ys[i]-my)**2}return num/Math.sqrt(dx*dy);}
function interpretations(obj){let rows=obj.rows,s=obj.summary,last=rows[rows.length-1],maxVol=rows.reduce((a,b)=>a.vol>b.vol?a:b),maxVolRatio=maxVol.vol/s.avg_vol;let bestM=obj.monthly.reduce((a,b)=>a.return_pct>b.return_pct?a:b),worstM=obj.monthly.reduce((a,b)=>a.return_pct<b.return_pct?a:b);let c=corr(rows.map(r=>r.vol),rows.map(r=>r.pct_chg));document.getElementById("i-kline").textContent=`区间收盘价从 ${fmtNum(rows[0].close)} 到 ${fmtNum(last.close)}，累计 ${pct(s.close_change_pct)}。最新收盘价 ${last.close>=last.ma20?"高于":"低于"} 20 日均线，短期趋势可结合 5 日均线斜率继续观察。`;document.getElementById("i-volume").textContent=`最高成交量出现在 ${fmtDate(maxVol.trade_date)}，约为平均成交量的 ${maxVolRatio.toFixed(1)} 倍，说明该日交易活跃度显著放大。`;document.getElementById("i-drawdown").textContent=`区间最大回撤为 ${pct(s.max_drawdown_pct)}，用于衡量从阶段高点回落的压力。回撤越深，说明持有过程中的波动承受要求越高。`;document.getElementById("i-hist").textContent=`上涨交易日 ${s.up_days} 天，下跌交易日 ${s.down_days} 天，胜率 ${s.up_ratio_pct.toFixed(1)}%。日收益中位数为 ${pct(s.median_return_pct)}。`;document.getElementById("i-volatility").textContent=`20 日滚动波动率最新值为 ${last.volatility20.toFixed(2)}%，若曲线抬升，通常意味着短期价格不确定性增加。`;document.getElementById("i-amount").textContent=`平均成交额为 ${fmtNum(s.avg_amount)}，成交额峰值常对应重要事件、情绪切换或趋势加速日。`;document.getElementById("i-scatter").textContent=`成交量与日涨跌幅的相关系数约为 ${c.toFixed(2)}。接近 0 表示价量线性关系不强，正值表示放量更偏向上涨日。`;document.getElementById("i-monthly").textContent=`表现最强月份为 ${bestM.month}（${pct(bestM.return_pct)}），最弱月份为 ${worstM.month}（${pct(worstM.return_pct)}），月度柱状图可用于识别趋势集中发生的阶段。`;document.getElementById("i-table").textContent=`关键交易日帮助定位异常波动：最大涨跌幅看价格冲击，最高成交量看资金关注度。`;}
function table(obj){let s=obj.summary;document.getElementById("keyTable").innerHTML=`<table class="table"><thead><tr><th>类型</th><th>日期</th><th>数值</th><th>收盘价</th></tr></thead><tbody><tr><td>最大涨幅</td><td>${fmtDate(s.best_day.date)}</td><td>${pct(s.best_day.pct_chg)}</td><td>${fmtNum(s.best_day.close)}</td></tr><tr><td>最大跌幅</td><td>${fmtDate(s.worst_day.date)}</td><td>${pct(s.worst_day.pct_chg)}</td><td>${fmtNum(s.worst_day.close)}</td></tr><tr><td>最高成交量</td><td>${fmtDate(s.highest_vol_day.date)}</td><td>${fmtNum(s.highest_vol_day.vol)}</td><td>${fmtNum(s.highest_vol_day.close)}</td></tr></tbody></table>`;}
function render(){let obj=DATA[current],s=obj.summary,rows=obj.rows;document.querySelectorAll(".tab").forEach(b=>b.classList.toggle("active",b.dataset.market===current));document.getElementById("metrics").innerHTML=[["市场",current+" / "+obj.code],["交易日",s.rows+" 天"],["累计涨跌",pct(s.close_change_pct)],["最大回撤",pct(s.max_drawdown_pct)],["上涨占比",s.up_ratio_pct.toFixed(1)+"%"],["最新收盘",fmtNum(s.last_close)]].map(x=>`<div class="metric"><div class="k">${x[0]}</div><div class="v">${x[1]}</div></div>`).join("");document.getElementById("csvLink").href=obj.csv;document.getElementById("csvName").textContent=obj.csv;drawK(rows);drawVolume(rows);drawCloseDrawdown(rows);drawHist(rows);drawSimpleLine("volatility",rows,"volatility20","#d97706");drawSimpleLine("amount",rows,"amount","#2563eb");drawScatter(rows);drawMonthly(obj);drawCompare();table(obj);interpretations(obj);let a=DATA["A股"].summary.close_change_pct,h=DATA["港股"].summary.close_change_pct;document.getElementById("i-compare").textContent=`按首日收盘价归一化后，A 股区间累计 ${pct(a)}，港股区间累计 ${pct(h)}。该图比较的是相对收益，不比较两地绝对价格。`;}
document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>{current=b.dataset.market;render();});window.addEventListener("resize",render);render();
</script>
</body>
</html>"""
    return (
        template.replace("__DATA__", json.dumps(data, ensure_ascii=False))
        .replace("__SUMMARIES__", json.dumps(summaries, ensure_ascii=False))
    )


def main():
    a_rows = enrich_rows(read_csv(BASE / "smic_a_daily.csv"))
    hk_rows = enrich_rows(read_csv(BASE / "smic_hk_daily.csv"))
    summaries = [
        summary_for("A股", "688981.SH", "smic_a_daily.csv", a_rows),
        summary_for("港股", "00981.HK", "smic_hk_daily.csv", hk_rows),
    ]
    by_market = {item["market"]: item for item in summaries}
    data = {
        "A股": {
            "code": "688981.SH",
            "csv": "smic_a_daily.csv",
            "rows": a_rows,
            "monthly": monthly_returns(a_rows),
            "summary": by_market["A股"],
        },
        "港股": {
            "code": "00981.HK",
            "csv": "smic_hk_daily.csv",
            "rows": hk_rows,
            "monthly": monthly_returns(hk_rows),
            "summary": by_market["港股"],
        },
    }
    (BASE / "smic_dashboard.html").write_text(build_html(data, summaries), encoding="utf-8")
    (BASE / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "html": str(BASE / "smic_dashboard.html"),
                "summary": summaries,
                "charts": 9,
                "tables": 1,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
