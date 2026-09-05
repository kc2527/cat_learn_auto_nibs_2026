from imports import *
from util_func_dbm import *


REQUIRED_BEHAV_COLS = {
    "subject_id",
    "session_num",
    "session_part",
    "trial",
    "phase",
    "cat",
    "x",
    "y",
    "xt",
    "yt",
    "resp_key",
    "resp",
    "fb",
    "rt",
    "ts_iso",
}


def read_behavior_csv(path):
    df = pd.read_csv(path)
    missing = REQUIRED_BEHAV_COLS.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )
    return df


def load_behavior_data(dir_data):
    records = []
    for root, _, files in os.walk(dir_data):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            path = os.path.join(root, fn)
            df = read_behavior_csv(path)
            df["source_path"] = path
            df["source_stream"] = (
                "lab_behave" if os.path.basename(root) == "behave" else "home"
            )
            records.append(df)

    if not records:
        raise FileNotFoundError(f"No behavior CSV files found under: {dir_data}")

    return pd.concat(records, ignore_index=True)


def prepare_behavior_frame(d, block_size=25):
    d = d.copy()
    d["session_num"] = pd.to_numeric(d["session_num"], errors="coerce")
    d = d.dropna(subset=["session_num"]).reset_index(drop=True)
    d["session_num"] = d["session_num"].astype(int)
    d = d.sort_values(["subject_id", "session_num", "trial"]).reset_index(drop=True)
    d["trial"] = d.groupby(["subject_id", "session_num"]).cumcount()
    d["n_trials"] = d.groupby(["subject_id", "session_num"])["trial"].transform("count")
    d["block"] = d.groupby(["subject_id", "session_num"])["trial"].transform(
        lambda x: x // block_size
    )
    d["acc"] = (d["cat"] == d["resp"]).astype(int)

    d["session_type"] = "Training at home"
    d.loc[d["source_stream"] == "lab_behave", "session_type"] = "Training in the Lab"
    d.loc[(d["source_stream"] == "home") & (d["session_num"] == 22),
          "session_type"] = "Dual-Task at home"
    d.loc[(d["source_stream"] == "home") & (d["session_num"].isin([23, 24])),
          "session_type"] = "Button-Switch at home"
    return d


if __name__ == "__main__":
    sns.set_palette("rocket", n_colors=4)
    plt.rc("axes", labelsize=14)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)

    dir_data = "../data"
    os.makedirs("../figures", exist_ok=True)
    os.makedirs("../dbm_fits", exist_ok=True)

    d_all = load_behavior_data(dir_data)
    d_all = prepare_behavior_frame(d_all, block_size=25)

    if {"ns_correct_side", "ns_resp"}.issubset(d_all.columns):
        d_all["acc_stroop"] = np.nan
        d_all.loc[d_all["ns_correct_side"].notna(), "acc_stroop"] = (
            d_all["ns_correct_side"] == d_all["ns_resp"]
        ).astype(int)
        d_all["acc_stroop_mean"] = d_all.groupby("subject_id")["acc_stroop"].transform(
            lambda x: np.nanmean(x)
        )
        d_all = d_all[d_all["acc_stroop_mean"] >= 0.8].reset_index(drop=True)

    d_train = d_all[d_all["session_type"] == "Training at home"].reset_index(drop=True)

    models = [nll_unix, nll_unix, nll_uniy, nll_uniy, nll_glc, nll_glc]
    side = [0, 1, 0, 1, 0, 1, 0, 1, 2, 3]
    k = [2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    model_names = [
        "nll_unix_0",
        "nll_unix_1",
        "nll_uniy_0",
        "nll_uniy_1",
        "nll_glc_0",
        "nll_glc_1",
    ]

    dbm_path = "../dbm_fits/dbm_results.csv"
    if not os.path.exists(dbm_path):
        dbm = (
            d_train.groupby(["subject_id", "session_num"])
            .apply(fit_dbm, models, side, k, 25, model_names)
            .reset_index()
        )
        dbm.to_csv(dbm_path, index=False)
    else:
        dbm = pd.read_csv(dbm_path)
        dbm = dbm[["subject_id", "session_num", "model", "bic", "p"]]

    def assign_best_model(x):
        model = x["model"].to_numpy()
        bic = x["bic"].to_numpy()
        x["best_model"] = np.unique(model[bic == bic.min()])[0]
        return x

    dbm = (
        dbm.groupby(["subject_id", "session_num"])
        .apply(assign_best_model, include_groups=False)
        .reset_index()
    )
    dbm = dbm[dbm["model"] == dbm["best_model"]]
    dbm = dbm[["subject_id", "session_num", "bic", "best_model"]].drop_duplicates()
    dbm["best_model_class"] = dbm["best_model"].str.split("_").str[1]
    dbm.loc[dbm["best_model_class"] != "glc", "best_model_class"] = "rule-based"
    dbm.loc[dbm["best_model_class"] == "glc", "best_model_class"] = "procedural"
    dbm["best_model_class"] = dbm["best_model_class"].astype("category")

    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 7))
    sns.pointplot(
        data=dbm,
        x="session_num",
        y="bic",
        hue="best_model_class",
        errorbar=("se"),
        ax=ax[0, 0],
    )
    ax[0, 0].set_xlabel("Session")
    ax[0, 0].set_ylabel("BIC")
    plt.tight_layout()
    plt.savefig("../figures/dbm_bic_performance.png", dpi=300)
    plt.close()

    d_plot = d_all[d_all["rt"] <= 3000]
    dd_all = (
        d_plot.groupby(["subject_id", "session_num", "session_type"])
        .agg({"acc": "mean", "rt": "mean"})
        .reset_index()
    )

    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 5))
    sns.pointplot(
        data=dd_all,
        x="session_num",
        y="acc",
        hue="session_type",
        errorbar=("se"),
        ax=ax[0, 0],
    )
    ax[0, 0].set_xlabel("Session", fontsize=16)
    ax[0, 0].set_ylabel("Proportion correct", fontsize=16)
    ax[0, 0].legend(title="")
    plt.savefig("../figures/training_performance_sessions.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 5))
    sns.pointplot(
        data=dd_all,
        x="session_num",
        y="rt",
        hue="session_type",
        errorbar=("se"),
        ax=ax[0, 0],
    )
    ax[0, 0].set_xlabel("Session", fontsize=16)
    ax[0, 0].set_ylabel("Reaction Time", fontsize=16)
    ax[0, 0].legend(title="")
    plt.savefig("../figures/training_rt_sessions.png", dpi=300)
    plt.close()

    d_dtf = dd_all[dd_all["session_num"].isin([20, 22])].copy()
    d_dtf["session_num"] = d_dtf["session_num"].map(
        {20: "Last Training Session", 22: "Dual-Task Session"}
    )
    fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(5, 8))
    sns.pointplot(data=d_dtf, x="session_num", y="acc", errorbar=("se"), ax=ax[0, 0])
    sns.pointplot(data=d_dtf, x="session_num", y="rt", errorbar=("se"), ax=ax[1, 0])
    ax[0, 0].set_xlabel("")
    ax[0, 0].set_ylabel("Accuracy (proportion correct)")
    ax[1, 0].set_xlabel("")
    ax[1, 0].set_ylabel("Reaction Time (ms)")
    plt.tight_layout()
    plt.savefig("../figures/dual_task_performance.png", dpi=300)
    plt.close()

    d_bsf = dd_all[dd_all["session_num"].isin([20, 23, 24])].copy()
    d_bsf["session_num"] = d_bsf["session_num"].map(
        {
            20: "Last Training Session",
            23: "Button-Switch Session 1",
            24: "Button-Switch Session 2",
        }
    )
    fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(7, 8))
    sns.pointplot(data=d_bsf, x="session_num", y="acc", errorbar=("se"), ax=ax[0, 0])
    sns.pointplot(data=d_bsf, x="session_num", y="rt", errorbar=("se"), ax=ax[1, 0])
    ax[0, 0].set_xlabel("")
    ax[0, 0].set_ylabel("Accuracy (proportion correct)")
    ax[1, 0].set_xlabel("")
    ax[1, 0].set_ylabel("Reaction Time (ms)")
    plt.tight_layout()
    plt.savefig("../figures/button_switch_performance.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 5))
    x = np.linspace(0, 1500, 1000)
    for i in range(5):
        y1 = (2 * i + 1) * np.exp(-0.5 * ((x - 500) / 100) ** 2)
        y2 = (i + 2) * np.exp(-0.5 * ((x - 1000) / 100) ** 2)
        ax[0, 0].plot(x, y1 + y2, label=f"Set {i + 1}")
    ax[0, 0].set_xlabel("Time within trial (ms)", fontsize=16)
    ax[0, 0].set_ylabel("Model signal (a.u.)", fontsize=16)
    ax[0, 0].legend().remove()
    ax[0, 0].legend([f"Session {i + 1}" for i in range(5)], title="")
    plt.savefig("../figures/model_predictions.png", dpi=300)
    plt.close()
