"""Semester at the study office: who should the advisers talk to at the end of week 6?

A Streamlit app for the study office (not for data scientists). It uses the logistic regression trained in the
notebook on the 2023-2024 cohorts, stored as plain JSON in model/ and read by portable_logit.py.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import portable_logit

st.set_page_config(page_title="Study office · week 6", page_icon="🎓", layout="wide")

URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"

# plain-language names for the model's inputs
LABELS = {"quiz_mean": "average quiz score", "fees_owed": "owes fees", "missed_last3": "assignments missed (last 3 weeks)",
          "logins_last3": "logins (last 3 weeks)", "logins_total": "logins (weeks 1-6)", "logins_trend": "change in logins",
          "submitted_share": "assignments handed in", "weeks_since_login": "weeks since last login",
          "su_scholarship": "SU grant", "age": "age", "admission_grade": "admission grade", "programme": "programme",
          "gender": "gender", "international": "international student", "first_gen": "first in family at university",
          "moved_from_home": "moved from home", "married": "married", "evening_programme": "evening programme"}
YES_NO = {"fees_owed", "su_scholarship", "international", "first_gen", "moved_from_home", "married", "evening_programme"}
# what a conversation could be about, by main reason
TOPIC = {"quiz_mean": "academic help", "missed_last3": "academic help", "submitted_share": "academic help",
         "admission_grade": "academic help", "fees_owed": "money: fees / SU grant", "su_scholarship": "money: fees / SU grant",
         "logins_last3": "check-in: how is it going?", "logins_total": "check-in: how is it going?",
         "logins_trend": "check-in: how is it going?", "weeks_since_login": "personal call: not seen lately"}
CANNOT_CHANGE = {"age", "programme", "gender", "international", "married", "first_gen", "moved_from_home", "evening_programme"}


# ---------- data and model (loaded once) ----------
@st.cache_resource
def load_model():
    return portable_logit.Model("model")


@st.cache_data
def load_data():
    history = pd.read_csv(URL + "history_week6.csv")
    new = pd.read_csv(URL + "new_week6.csv")
    return history[history["cohort"] == 2025].copy(), new


model = load_model()
past, new = load_data()
past["risk"] = model.predict_proba(past)
new["risk"] = model.predict_proba(new)
push_new = model.contributions(new)
new["main_reason"] = push_new.idxmax(axis=1)
typical = new[model.features].median(numeric_only=True)


def show(col, value):
    if col in YES_NO:
        return "yes" if value == 1 else "no"
    if col == "submitted_share":
        return f"{value:.0%}"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "–" if pd.isna(value) else f"{value:.0f}"
    return str(value)


def top_k(data, k):
    """Mark the k highest risks in data (the office's rule)."""
    return data["risk"].rank(ascending=False, method="first") <= k


def boxes(data, contacted):
    left = data["left"] == 1
    tp = int((contacted & left).sum()); fp = int((contacted & ~left).sum())
    fn = int((~contacted & left).sum()); tn = int((~contacted & ~left).sum())
    return {"contacted": tp + fp, "reached": tp, "worried": fp, "missed": fn, "fine": tn,
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}


# ---------- sidebar: the rule and the costs ----------
st.sidebar.header("The rule")
k = st.sidebar.slider("Conversations the advisers can hold", 10, 150, 40, 1,
                      help="Three advisers can hold about 40 conversations at the end of week 6.")
st.sidebar.caption("The rule: talk to the students with the highest risk, as many as there are conversations.")

st.sidebar.header("What the mistakes cost (DKK)")
st.sidebar.caption("These are assumptions, a policy choice. Change them and see how the best rule moves.")
c_talk = st.sidebar.number_input("One conversation (adviser time)", 0, 10_000, 500, 100)
c_worry = st.sidebar.number_input("One student worried for nothing", 0, 50_000, 2_000, 500)
c_leave = st.sidebar.number_input("One student who leaves", 0, 500_000, 60_000, 5_000)
helps = st.sidebar.slider("Share of students at risk a conversation keeps", 0.0, 1.0, 0.30, 0.05)

# ---------- header ----------
st.title("🎓 Semester at the study office")
st.markdown(
    "At the end of week 6 the advisers can hold about **40 conversations**. This tool ranks this year's students "
    "by their **risk of leaving later this semester**, so the advisers can decide whom to invite.  \n"
    "**The tool does not decide anything.** It offers a list; an adviser looks at every name and decides. "
    "Students are told that the office uses it, and they can ask not to be included.")

tab_list, tab_rule, tab_groups, tab_why = st.tabs(
    ["📋 This week's list", "⚖️ What the rule gets wrong", "🌍 Fair to whom?", "🔍 Why is a student on the list?"])

# ---------- 1 · this week's list ----------
with tab_list:
    new["on_list"] = top_k(new, k)
    on = new[new["on_list"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Students this week", f"{len(new)}")
    c2.metric("Invited for a conversation", f"{k}")
    c3.metric("Expected to leave among them (estimate)", f"≈ {on['risk'].sum():.0f}")
    st.caption(f"The lowest risk on the list is {on['risk'].min():.0%}. On 2025 the model was slightly too "
               "pessimistic, so the real number may be a little lower.")

    show_all = st.toggle("Show all students, not only those on the list", value=False)
    table = (new if show_all else on).sort_values("risk", ascending=False)
    view = pd.DataFrame({
        "rank": range(1, len(table) + 1),
        "student": table["student_id"].values,
        "on the list": np.where(table["on_list"], "✅", ""),
        "risk of leaving": (table["risk"] * 100).round(0).astype(int).astype(str).values + " %",
        "main reason": [LABELS[r] for r in table["main_reason"]],
        "suggested conversation": [TOPIC.get(r, "adviser decides first (not a study reason)")
                                   for r in table["main_reason"]],
        "programme": table["programme"].values,
        "international": np.where(table["international"] == 1, "yes", "no"),
    })
    st.dataframe(view, hide_index=True, width="stretch", height=min(38 * (len(view) + 1), 600))
    n_cannot = int(on["main_reason"].isin(CANNOT_CHANGE).sum())
    if n_cannot:
        st.warning(f"{n_cannot} of the {k} students are on the list mainly because of something they cannot change "
                   "(age, programme, ...). An adviser should check their study activity before inviting them.")
    st.download_button("Download the list (CSV)", view.to_csv(index=False).encode(), "week6_list.csv", "text/csv")

# ---------- 2 · what the rule gets wrong (on 2025, where the outcome is known) ----------
with tab_rule:
    st.subheader(f"If the office had held {k} conversations in 2025 …")
    st.caption("We test the rule on the 2025 students: the model never saw them, and we know who left.")
    b = boxes(past, top_k(past, k))
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Reached in time", b["reached"], help="Invited, and would have left: the conversation came in time.")
    c2.metric("📞 Worried for nothing", b["worried"], help="Invited, but would have stayed anyway.")
    c3.metric("🚪 Missed", b["missed"], help="Not invited, and left.")
    st.markdown(
        f"Of the **{b['contacted']} students invited**, **{b['reached']}** really were at risk "
        f"(**{b['precision']:.0%}**, precision). Of the **{b['reached'] + b['missed']} students who left**, "
        f"the office would have reached **{b['reached']}** (**{b['recall']:.0%}**, recall). "
        f"Picking {k} students at random would have reached about {k * past['left'].mean():.0f}.")

    # every possible number of conversations: what it reaches and what it is worth
    ks = np.arange(5, 301, 5)
    curve = pd.DataFrame([boxes(past, top_k(past, n)) | {"conversations": n} for n in ks])
    curve["net value (DKK)"] = (curve["reached"] * helps * c_leave - curve["contacted"] * c_talk
                                - curve["worried"] * c_worry)
    best = curve.loc[curve["net value (DKK)"].idxmax()]
    break_even = (c_talk + c_worry) / max(1, helps * c_leave + c_worry)

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("**More conversations reach more students, but each one reaches fewer**")
        long = curve.melt("conversations", ["reached", "missed", "worried"], "outcome", "students")
        long["outcome"] = long["outcome"].map({"reached": "reached in time", "missed": "missed",
                                               "worried": "worried for nothing"})
        chart = alt.Chart(long).mark_line(strokeWidth=2).encode(
            x=alt.X("conversations:Q", title="conversations"), y=alt.Y("students:Q", title="students (2025)"),
            color=alt.Color("outcome:N", title=None,
                            scale=alt.Scale(domain=["reached in time", "missed", "worried for nothing"],
                                            range=["#1baf7a", "#e34948", "#9aa0a6"])),
            tooltip=["conversations", "outcome", "students"])
        rule = alt.Chart(pd.DataFrame({"x": [k]})).mark_rule(strokeDash=[4, 4], color="#555").encode(x="x:Q")
        st.altair_chart(chart + rule, width="stretch")
    with right_col:
        st.markdown("**What each number of conversations is worth, with your costs**")
        value = alt.Chart(curve).mark_line(strokeWidth=2, color="#2a78d6").encode(
            x=alt.X("conversations:Q", title="conversations"),
            y=alt.Y("net value (DKK):Q", title="net value on 2025 (DKK)"),
            tooltip=["conversations", alt.Tooltip("net value (DKK):Q", format=",.0f")])
        st.altair_chart(value + rule, width="stretch")

    now = curve.loc[(curve["conversations"] - k).abs().idxmin(), "net value (DKK)"]
    st.info(
        f"With your costs, a conversation pays off for a student with a risk above **{break_even:.1%}**. "
        f"On 2025 the most valuable choice would have been about **{int(best['conversations'])} conversations** "
        f"(≈ {best['net value (DKK)']:,.0f} DKK), against ≈ {now:,.0f} DKK with {k}. "
        "These costs are assumptions: who decides what a worried student 'costs' is a question for people, "
        "not for the model.")

# ---------- 3 · fair to whom? ----------
with tab_groups:
    st.subheader(f"The same rule ({k} conversations), for each group of 2025 students")
    contacted = top_k(past, k)
    rows = []
    for name, mask in [("domestic", past["international"] == 0), ("international", past["international"] == 1)]:
        g = boxes(past[mask], contacted[mask])
        rows.append({"group": name, "students": int(mask.sum()),
                     "really left": f"{past.loc[mask, 'left'].mean():.1%}",
                     "predicted risk (average)": f"{past.loc[mask, 'risk'].mean():.1%}",
                     "invited": g["contacted"], "reached in time": g["reached"], "missed": g["missed"],
                     "worried for nothing": g["worried"],
                     "of those who left, share reached": f"{g['recall']:.0%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(
        "**How to read this.** Both groups leave about equally often, but the model does not treat them the same: "
        "international students who leave are reached less often. One reason: international students who *stay* "
        "log in about as rarely as domestic students who *leave*, so a login is not the same signal for everyone.  \n"
        "Only about 14 international students left in 2025, so the numbers move a lot with one or two students. "
        "Whether both groups should be reached equally often is a decision for the university, made openly.")

# ---------- 4 · why is a student on the list? ----------
with tab_why:
    ranked = new.sort_values("risk", ascending=False)
    options = [f"{r.student_id} · {r.risk:.0%}{' · on the list' if r.on_list else ''}" for r in ranked.itertuples()]
    choice = st.selectbox("Student (highest risk first)", options)
    sid = choice.split(" · ")[0]
    row = new[new["student_id"] == sid]
    s = row.iloc[0]
    push = push_new.loc[row.index[0]].sort_values()

    st.markdown(f"### {sid} · {s['programme']} · risk of leaving **{s['risk']:.0%}**")
    up_col, down_col = st.columns(2)
    with up_col:
        st.markdown("**What raises the risk most**")
        for col in push[push > 0.05].index[::-1][:5]:
            typ = f" (typical student: {show(col, typical[col])})" if col in typical else ""
            st.markdown(f"↑ {LABELS[col]}: **{show(col, s[col])}**{typ}")
    with down_col:
        st.markdown("**What lowers the risk**")
        for col in push[push < -0.05].index[:3]:
            typ = f" (typical student: {show(col, typical[col])})" if col in typical else ""
            st.markdown(f"↓ {LABELS[col]}: **{show(col, s[col])}**{typ}")

    keep = list(push.head(3).index) + list(push.tail(5).index)
    bars = push[push.index.isin(keep) & (push.abs() > 0.05)].rename(index=LABELS).reset_index()
    bars.columns = ["factor", "push"]
    bars["direction"] = np.where(bars["push"] > 0, "raises the risk", "lowers the risk")
    chart = alt.Chart(bars).mark_bar(cornerRadius=3).encode(
        x=alt.X("push:Q", title="← lowers the risk          raises the risk →"),
        y=alt.Y("factor:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.Color("direction:N", legend=None,
                        scale=alt.Scale(domain=["raises the risk", "lowers the risk"], range=["#eb6834", "#2a78d6"])))
    st.altair_chart(chart, width="stretch")
    if s["main_reason"] in CANNOT_CHANGE:
        st.warning("This student is on the list mainly because of something they cannot change. "
                   "Check their study activity before inviting them.")
    st.caption("The model reads all factors together, so related factors (quiz scores, assignments, logins) share "
               "the effect between them. The long bars matter; short bars are not advice. "
               "Suggested opening: \"We talk to many first-year students around now: how is it going?\" "
               "never \"our system thinks you will drop out\".")

st.divider()
st.caption("Model: logistic regression, trained on the 2023–2024 cohorts, checked on 2025 (AUC 0.81). "
           "Data: synthetic students resampled from the UCI 'Predict students' dropout and academic success' data "
           "(CC BY 4.0). Before real use: retrain on all years, check every new cohort, and keep a person in charge.")
