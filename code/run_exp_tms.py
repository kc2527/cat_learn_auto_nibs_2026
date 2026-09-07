# -*- coding: utf-8 -*-
"""
Run lab-day TMS category learning experiment using PsychoPy.
"""

from datetime import datetime, timedelta
import hashlib
import os
import sys
import numpy as np
import pandas as pd
from psychopy import visual, core
from psychopy.hardware import keyboard
from util_func_tms import TriggerPort
from util_func_pid import prompt_for_pid
from util_func_session_man import resolve_session
from util_func_stimcat import make_stim_cats
from util_func_stimcat import plot_stim_space_examples
from util_func_stimcat import stim_xy_to_sf_ori_deg
from util_func_stimcat import transform_stim

TMS_TRIGGER_ENABLED = False
TMS_TRIGGER_PORT_ADDRESS = '0x3FD8'
TMS_TRIGGER_DEFAULT_PULSE_MS = 10
STUDY_TAG = "pace_tms_2026"

TRIG = {
    # Bistim2 TMS device trigger (bit 0).
    "BISTIM": 1,

    # Rapid2 TMS device trigger (bit 1).
    "RAPID": 2,

    # LabChart label sent with a Bistim2 TMS pulse (bit 2).
    "BISTIM_LABEL": 4,

    # LabChart label sent with a Rapid2 TMS pulse (bit 3).
    "RAPID_LABEL": 8,

    # Experiment handling (bit 4).
    "EXP_START": 16,
    "ITI_ONSET": 16,
    "EXP_END": 16,

    # Task stimulus onset (bit 5).
    "STIM_ONSET_A": 32,
    "STIM_ONSET_B": 32,

    # Task response (bit 6).
    "RESP_A": 64,
    "RESP_B": 64,

    # Task feedback (bit 7).
    "FB_COR": 128,
    "FB_INC": 128,
}

V1_REST_BLOCK = {
    "name": "V1 to M1 rest",
    "kind": "rest",
    "site": "V1",
    "n_trials": 50,
    "setup_text": "Rapid2 coil: V1\nBistim2 coil: M1",
}

V1_TASK_BLOCK = {
    "name": "V1 to M1 task",
    "kind": "task",
    "site": "V1",
    "n_trials": 100,
    "setup_text": "Rapid2 coil: V1\nBistim2 coil: M1",
}

VERTEX_REST_BLOCK = {
    "name": "Vertex to M1 rest",
    "kind": "rest",
    "site": "Vertex",
    "n_trials": 50,
    "setup_text": "Rapid2 coil: Vertex\nBistim2 coil: M1",
}

VERTEX_TASK_BLOCK = {
    "name": "Vertex to M1 task",
    "kind": "task",
    "site": "Vertex",
    "n_trials": 100,
    "setup_text": "Rapid2 coil: Vertex\nBistim2 coil: M1",
}

M1_ICI_BLOCK = {
    "name": "M1 to M1 ICI rest",
    "kind": "m1_ici",
    "site": "M1",
    "n_trials": 30,
    "setup_text": "Bistim2 coil: M1\nSet Bistim interval: 5 ms\nRapid2: not used",
}

M1_ICF_BLOCK = {
    "name": "M1 to M1 ICF rest",
    "kind": "m1_icf",
    "site": "M1",
    "n_trials": 30,
    "setup_text": "Bistim2 coil: M1\nSet Bistim interval: 15 ms\nRapid2: not used",
}

M1_TEST_BLOCK = {
    "name": "M1 single-pulse rest",
    "kind": "m1_test",
    "site": "M1",
    "n_trials": 30,
    "setup_text": "Bistim2 coil: M1\nSet Bistim to single-pulse mode\nRapid2: not used",
}

PID_DIGITS = 3
BLOCK_ORDER_BY_SUBJECT = {
    "002": "v1_first",
    "077": "vertex_first",
    "134": "v1_first",
    "189": "vertex_first",
    "213": "v1_first",
    "268": "vertex_first",
    "303": "v1_first",
    "358": "vertex_first",
    "482": "v1_first",
    "527": "vertex_first",
    "594": "v1_first",
    "639": "vertex_first",
    "662": "v1_first",
    "707": "vertex_first",
    "729": "v1_first",
    "875": "vertex_first",
    "943": "v1_first",
    "998": "vertex_first",
}

PIXELS_PER_INCH = 227 / 2
PX_PER_CM = PIXELS_PER_INCH / 2.54
SIZE_CM = 5
SIZE_PX = int(SIZE_CM * PX_PER_CM)
RESUME_WINDOW = timedelta(hours=12)
NEW_SESSION_COOLDOWN = timedelta(hours=8)

# ----------------------------------------------------------------------------------


def stable_int_seed(label):
    digest = hashlib.sha256(str(label).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


if __name__ == "__main__":

    # --------------------------- Display / geometry -------------------------------

    win = visual.Window(
        size=(1920, 1080),
        fullscr=True,
        units='pix',
        color=(0.494, 0.494, 0.494),
        colorSpace='rgb',
        winType='pyglet',
        useRetina=True,
        waitBlanking=True,
    )
    win.mouseVisible = False

    # --------------------------- Stim objects ------------------------------------
    fix_h = visual.Line(win,
                        start=(0, -10),
                        end=(0, 10),
                        lineColor='white',
                        lineWidth=8)
    fix_v = visual.Line(win,
                        start=(-10, 0),
                        end=(10, 0),
                        lineColor='white',
                        lineWidth=8)

    init_text = visual.TextStim(win,
                                text="Please press the space bar to begin",
                                color='white',
                                height=32)

    finished_text = visual.TextStim(
        win,
        text="You finished! Thank you for participating!",
        color='white',
        height=32)

    break_text = visual.TextStim(
        win,
        text="",
        color='white',
        height=32,
        wrapWidth=1400)

    grating = visual.GratingStim(win,
                                 tex='sin',
                                 mask='circle',
                                 texRes=256,
                                 interpolate=True,
                                 size=(SIZE_PX, SIZE_PX),
                                 units='pix',
                                 sf=0.0,
                                 ori=0.0)

    fb_ring = visual.Circle(win,
                            radius=(SIZE_PX // 2 + 10),
                            edges=128,
                            fillColor=None,
                            lineColor='white',
                            lineWidth=10,
                            units='pix',
                            pos=(0, 0))

    # --------------------------- response and clocks -----------------------------
    kb = keyboard.Keyboard()
    default_kb = keyboard.Keyboard()

    global_clock = core.Clock()
    state_clock = core.Clock()
    stim_clock = core.Clock()

    # --------------------------- Subject handling --------------------------------
    dir_data = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(dir_data, exist_ok=True)

    subject, block_order = prompt_for_pid(
        win, PID_DIGITS, BLOCK_ORDER_BY_SUBJECT)

    if block_order == "v1_first":
        tms_blocks = [
            V1_REST_BLOCK,
            V1_TASK_BLOCK,
            VERTEX_REST_BLOCK,
            VERTEX_TASK_BLOCK,
            M1_ICI_BLOCK,
            M1_ICF_BLOCK,
            M1_TEST_BLOCK,
        ]
    else:
        tms_blocks = [
            VERTEX_REST_BLOCK,
            VERTEX_TASK_BLOCK,
            V1_REST_BLOCK,
            V1_TASK_BLOCK,
            M1_ICI_BLOCK,
            M1_ICF_BLOCK,
            M1_TEST_BLOCK,
        ]

    n_total = sum(block["n_trials"] for block in tms_blocks)

    # ---------------------------  session handling -------------------------------
    session_info = resolve_session(
        dir_data,
        subject,
        n_total,
        resume_window=RESUME_WINDOW,
        new_session_cooldown=NEW_SESSION_COOLDOWN,
        study_tag=STUDY_TAG,
    )
    session_num = session_info["session_num"]
    part_num = session_info["part_num"]
    today_key = session_info["today_key"]
    f_name = session_info["f_name"]
    full_path = session_info["full_path"]
    n_done = session_info["n_done"]

    # --------------------------- Trial schedule  ---------------------------------
    session_seed = stable_int_seed(f"{subject}_{session_num:03d}_tms")
    schedule_rng = np.random.default_rng(session_seed)
    task_stimuli, _, _ = make_stim_cats(100, random_seed=session_seed)
    task_stimuli = task_stimuli.sample(
        frac=1,
        random_state=int(schedule_rng.integers(0, 2**32 - 1)),
    ).reset_index(drop=True)

    trial_rows = []
    stim_index = 0
    for block_num, block in enumerate(tms_blocks, start=1):
        if block["kind"] == "rest":
            trial_types = ["conditioned"] * 25 + ["test_alone"] * 25
        elif block["kind"] == "task":
            trial_types = ["conditioned"] * 50 + ["test_alone"] * 50
        elif block["kind"] == "m1_ici":
            trial_types = ["ici"] * 30
        elif block["kind"] == "m1_icf":
            trial_types = ["icf"] * 30
        else:
            trial_types = ["test_alone"] * 30
        if len(set(trial_types)) > 1:
            schedule_rng.shuffle(trial_types)

        for trial_type in trial_types:
            row = {
                "block_num": block_num,
                "block_name": block["name"],
                "block_kind": block["kind"],
                "conditioning_site": block["site"],
                "trial_type": trial_type,
                "phase": "rest",
                "cat": "",
                "x": np.nan,
                "y": np.nan,
                "pulse_1_device": "",
                "pulse_1_ms": np.nan,
                "pulse_2_device": "",
                "pulse_2_ms": np.nan,
                "isi_ms": np.nan,
            }

            if block["kind"] == "rest":
                if trial_type == "conditioned":
                    row["pulse_1_device"] = "RAPID"
                    row["pulse_1_ms"] = 0
                    row["pulse_2_device"] = "BISTIM"
                    row["pulse_2_ms"] = 40
                    row["isi_ms"] = 40
                else:
                    row["pulse_1_device"] = "BISTIM"
                    row["pulse_1_ms"] = 40
            elif block["kind"] == "task":
                stim = task_stimuli.iloc[stim_index]
                stim_index += 1
                row["phase"] = "train"
                row["cat"] = str(stim["cat"]).upper()
                row["x"] = stim["x"]
                row["y"] = stim["y"]
                if trial_type == "conditioned":
                    row["pulse_1_device"] = "RAPID"
                    row["pulse_1_ms"] = 50
                    row["pulse_2_device"] = "BISTIM"
                    row["pulse_2_ms"] = 90
                    row["isi_ms"] = 40
                else:
                    row["pulse_1_device"] = "BISTIM"
                    row["pulse_1_ms"] = 90
            elif block["kind"] == "m1_ici":
                row["pulse_1_device"] = "BISTIM"
                row["pulse_1_ms"] = 0
                row["isi_ms"] = 5
            elif block["kind"] == "m1_icf":
                row["pulse_1_device"] = "BISTIM"
                row["pulse_1_ms"] = 0
                row["isi_ms"] = 15
            else:
                row["pulse_1_device"] = "BISTIM"
                row["pulse_1_ms"] = 0

            trial_rows.append(row)

    trials = pd.DataFrame(trial_rows)
    if len(trials) != n_total or stim_index != 200:
        raise RuntimeError("The laboratory trial schedule has the wrong size.")

    trial = n_done - 1

    # --------------------------- TMS init ----------------------------------------
    trigger_port = TriggerPort(
        win,
        address=TMS_TRIGGER_PORT_ADDRESS,
        enabled=TMS_TRIGGER_ENABLED,
        default_ms=TMS_TRIGGER_DEFAULT_PULSE_MS,
    )

    # --------------------------- State machine setup ------------------------------
    time_state = 0.0
    state_current = "state_init"
    state_entry = True

    resp_key = ""
    resp = ""
    fb = ""
    rt = -1
    trial = n_done - 1
    phase = ""
    cat = ""
    gap_ms = 0
    sf_cycles_per_pix = np.nan
    ori_deg = np.nan
    trig_stim = np.nan
    trig_resp = np.nan
    trig_fb = np.nan
    t_resp = np.nan
    current_trial = None
    pulse_1_sent = False
    pulse_2_sent = False
    pulse_1_code = np.nan
    pulse_2_code = np.nan

    # Record keeping
    trial_data = {
        "subject_id": [],
        "study_tag": [],
        "session_num": [],
        "session_part": [],
        "trial": [],
        "block_num": [],
        "block_name": [],
        "block_kind": [],
        "conditioning_site": [],
        "trial_type": [],
        "pulse_1_device": [],
        "pulse_1_ms": [],
        "pulse_2_device": [],
        "pulse_2_ms": [],
        "isi_ms": [],
        "phase": [],
        "cat": [],
        "resp_key": [],
        "resp": [],
        "fb": [],
        "rt": [],
        "ts_iso": [],
        "tms_trigger_enabled": [],
        "trigger_stim": [],
        "trigger_resp": [],
        "trigger_fb": [],
        "t_stim": [],
        "t_trial_onset": [],
        "trigger_pulse_1": [],
        "trigger_pulse_2": [],
        "t_pulse_1": [],
        "t_pulse_2": [],
        "t_resp": [],
        "t_fb": [],
        "port_address": [],
        "block_order": [],
        "x": [],
        "y": [],
        "xt": [],
        "yt": []
    }

    flip_times = {
        "t_stim": np.nan,
        "t_fb": np.nan,
        "t_trial_onset": np.nan,
        "t_pulse_1": np.nan,
        "t_pulse_2": np.nan,
    }

    # --------------------------- Main loop ---------------------------------------
    running = True
    while running:

        if default_kb.getKeys(keyList=['escape'], waitRelease=False):
            running = False
            break

        trigger_port.update(global_clock)

        # --------------------- STATE: INIT ---------------------
        if state_current == "state_init":
            if state_entry:
                state_clock.reset()
                win.color = (0.494, 0.494, 0.494)
                state_entry = False

            time_state = state_clock.getTime() * 1000.0
            init_text.draw()

            keys = kb.getKeys(keyList=['space'], waitRelease=False, clear=True)
            if keys:
                trigger_port.flip_pulse(TRIG["EXP_START"], global_clock=global_clock)
                state_current = "state_block_start"
                state_entry = True

            win.flip()

        # --------------------- STATE: FINISHED ---------------------
        elif state_current == "state_finished":
            if state_entry:
                trigger_port.flip_pulse(TRIG["EXP_END"], global_clock=global_clock)
                state_clock.reset()
                state_entry = False

            time_state = state_clock.getTime() * 1000.0
            finished_text.draw()
            win.flip()

        # --------------------- STATE: BLOCK START -------------
        elif state_current == "state_block_start":
            if state_entry:
                next_trial = trial + 1
                block_num = int(trials["block_num"].iloc[next_trial])
                block = tms_blocks[block_num - 1]
                break_text.text = (
                    "Remember to rearm TMS machines.\n\n"
                    f"Block {block_num}: {block['name']}\n\n"
                    f"{block['setup_text']}\n\n"
                    "Press the space bar when ready to begin."
                )
                state_entry = False

            break_text.draw()

            keys = kb.getKeys(keyList=['space'], waitRelease=False, clear=True)
            if keys:
                state_current = "state_iti"
                state_entry = True

            win.flip()

        # --------------------- STATE: BREAK -------------------
        elif state_current == "state_break":
            if state_entry:
                break_text.text = (
                    "Remember to rearm TMS machines.\n\n"
                    "Take a short break.\n\n"
                    "Press the space bar when ready to continue."
                )
                state_entry = False

            break_text.draw()

            keys = kb.getKeys(keyList=['space'], waitRelease=False, clear=True)
            if keys:
                state_current = "state_iti"
                state_entry = True

            win.flip()

        # --------------------- STATE: ITI ---------------------
        elif state_current == "state_iti":
            if state_entry:
                state_clock.reset()
                trigger_port.flip_pulse(TRIG["ITI_ONSET"], global_clock=global_clock)
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            fix_h.draw()
            fix_v.draw()

            if time_state >= 5000:
                resp_key = ""
                resp = ""
                fb = ""
                rt = -1
                t_resp = np.nan
                state_clock.reset()
                trial += 1
                if trial >= n_total:
                    state_current = "state_finished"
                    state_entry = True
                else:
                    current_trial = trials.iloc[trial]
                    phase = current_trial["phase"]
                    cat = current_trial["cat"]
                    trig_stim = np.nan
                    trig_resp = np.nan
                    trig_fb = np.nan
                    pulse_1_code = np.nan
                    pulse_2_code = np.nan
                    pulse_1_sent = False
                    pulse_2_sent = False
                    flip_times["t_stim"] = np.nan
                    flip_times["t_fb"] = np.nan
                    flip_times["t_trial_onset"] = np.nan
                    flip_times["t_pulse_1"] = np.nan
                    flip_times["t_pulse_2"] = np.nan

                    if phase == "train":
                        sf_cycles_per_pix, ori_deg = stim_xy_to_sf_ori_deg(
                            current_trial["x"],
                            current_trial["y"],
                            PX_PER_CM,
                        )
                        if cat not in {"A", "B"}:
                            raise ValueError(
                                f"Category labels must be 'A' or 'B'. Got: {cat}")
                        grating.sf = sf_cycles_per_pix
                        grating.ori = ori_deg
                        grating.pos = (0, 0)
                        kb.clearEvents()
                        state_current = "state_stim"
                    else:
                        state_current = "state_rest"
                    state_entry = True

            win.flip()

        # --------------------- STATE: REST ---------------------
        elif state_current == "state_rest":
            if state_entry:
                state_clock.reset()
                win.callOnFlip(state_clock.reset)
                win.callOnFlip(
                    lambda: flip_times.__setitem__(
                        "t_trial_onset", global_clock.getTime()))

                if current_trial["pulse_1_ms"] == 0:
                    pulse_1_code = (
                        TRIG[current_trial["pulse_1_device"]] |
                        TRIG[f"{current_trial['pulse_1_device']}_LABEL"]
                    )
                    trigger_port.flip_pulse(
                        pulse_1_code, global_clock=global_clock)
                    win.callOnFlip(
                        lambda: flip_times.__setitem__(
                            "t_pulse_1", global_clock.getTime()))
                    pulse_1_sent = True
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            fix_h.draw()
            fix_v.draw()

            if not pulse_1_sent and time_state >= current_trial["pulse_1_ms"]:
                pulse_1_code = (
                    TRIG[current_trial["pulse_1_device"]] |
                    TRIG[f"{current_trial['pulse_1_device']}_LABEL"]
                )
                trigger_port.pulse_now(pulse_1_code, global_clock=global_clock)
                flip_times["t_pulse_1"] = global_clock.getTime()
                pulse_1_sent = True

            if (current_trial["pulse_2_device"] and not pulse_2_sent and
                    time_state >= current_trial["pulse_2_ms"]):
                pulse_2_code = (
                    TRIG[current_trial["pulse_2_device"]] |
                    TRIG[f"{current_trial['pulse_2_device']}_LABEL"]
                )
                trigger_port.pulse_now(pulse_2_code, global_clock=global_clock)
                flip_times["t_pulse_2"] = global_clock.getTime()
                pulse_2_sent = True

            if pulse_1_sent and (not current_trial["pulse_2_device"] or pulse_2_sent):
                state_current = "state_save_trial"
                state_entry = True

            win.flip()

        # --------------------- STATE: STIM ---------------------
        elif state_current == "state_stim":
            if state_entry:
                if cat == "A":
                    trig = TRIG["STIM_ONSET_A"]
                else:
                    trig = TRIG["STIM_ONSET_B"]

                trigger_port.flip_pulse(trig, global_clock=global_clock)
                trig_stim = int(trig)

                state_clock.reset()
                stim_clock.reset()

                win.callOnFlip(lambda: flip_times.__setitem__("t_stim", global_clock.getTime()))
                win.callOnFlip(lambda: flip_times.__setitem__("t_trial_onset", global_clock.getTime()))
                win.callOnFlip(stim_clock.reset)
                win.callOnFlip(kb.clock.reset)
                state_entry = False

            time_state = stim_clock.getTime() * 1000.0

            grating.draw()

            if not pulse_1_sent and time_state >= current_trial["pulse_1_ms"]:
                pulse_1_code = (
                    TRIG[current_trial["pulse_1_device"]] |
                    TRIG[f"{current_trial['pulse_1_device']}_LABEL"]
                )
                trigger_port.pulse_now(pulse_1_code, global_clock=global_clock)
                flip_times["t_pulse_1"] = global_clock.getTime()
                pulse_1_sent = True

            if (current_trial["pulse_2_device"] and not pulse_2_sent and
                    time_state >= current_trial["pulse_2_ms"]):
                pulse_2_code = (
                    TRIG[current_trial["pulse_2_device"]] |
                    TRIG[f"{current_trial['pulse_2_device']}_LABEL"]
                )
                trigger_port.pulse_now(pulse_2_code, global_clock=global_clock)
                flip_times["t_pulse_2"] = global_clock.getTime()
                pulse_2_sent = True

            pulses_complete = (
                pulse_1_sent and
                (not current_trial["pulse_2_device"] or pulse_2_sent)
            )
            keys = kb.getKeys(keyList=['d', 'k'], waitRelease=False)
            if keys and pulses_complete:
                k = keys[-1]
                resp_key = k.name
                rt = k.rt * 1000.0
                if k.name == 'd':
                    resp_label = "A"
                else:
                    resp_label = "B"

                if resp_label == "A":
                    trig = TRIG["RESP_A"]
                else:
                    trig = TRIG["RESP_B"]

                trigger_port.pulse_now(trig, global_clock=global_clock)
                trig_resp = int(trig)
                t_resp = global_clock.getTime()

                if cat == resp_label:
                    fb = "Correct"
                else:
                    fb = "Incorrect"
                resp = resp_label

                state_clock.reset()
                gap_ms = np.random.randint(200, 401)
                state_current = "state_pre_feedback_gap"
                state_entry = True

            win.flip()

        # --------------------- STATE: PRE-FEEDBACK GAP ---------------------
        elif state_current == "state_pre_feedback_gap":
            if state_entry:
                state_clock.reset()
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            grating.draw()

            if time_state >= gap_ms:
                state_current = "state_feedback"
                state_entry = True

            win.flip()

        # --------------------- STATE: FEEDBACK ---------------------
        elif state_current == "state_feedback":
            if state_entry:
                if fb == "Correct":
                    fb_ring.lineColor = 'green'
                else:
                    fb_ring.lineColor = 'red'

                if phase == 'train':
                    if fb == "Correct":
                        trig = TRIG["FB_COR"]
                    else:
                        trig = TRIG["FB_INC"]
                else:
                    trig = np.nan

                if not np.isnan(trig):
                    trigger_port.flip_pulse(trig, global_clock=global_clock)
                    trig_fb = int(trig)

                win.callOnFlip(lambda: flip_times.__setitem__("t_fb", global_clock.getTime()))
                state_clock.reset()
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            grating.draw()
            fb_ring.draw()

            if time_state > 1000:
                state_current = "state_save_trial"
                state_entry = True

            win.flip()

        # --------------------- STATE: SAVE TRIAL ---------------
        elif state_current == "state_save_trial":
            if phase == "train":
                xt, yt = transform_stim(
                    current_trial["x"], current_trial["y"])
            else:
                xt = np.nan
                yt = np.nan

            trial_data["subject_id"].append(subject)
            trial_data["study_tag"].append(STUDY_TAG)
            trial_data["session_num"].append(session_num)
            trial_data["session_part"].append(part_num)
            trial_data["trial"].append(trial)
            trial_data["block_num"].append(current_trial["block_num"])
            trial_data["block_name"].append(current_trial["block_name"])
            trial_data["block_kind"].append(current_trial["block_kind"])
            trial_data["conditioning_site"].append(
                current_trial["conditioning_site"])
            trial_data["trial_type"].append(current_trial["trial_type"])
            trial_data["pulse_1_device"].append(current_trial["pulse_1_device"])
            trial_data["pulse_1_ms"].append(current_trial["pulse_1_ms"])
            trial_data["pulse_2_device"].append(current_trial["pulse_2_device"])
            trial_data["pulse_2_ms"].append(current_trial["pulse_2_ms"])
            trial_data["isi_ms"].append(current_trial["isi_ms"])
            trial_data["phase"].append(phase)
            trial_data["cat"].append(cat)
            trial_data["resp_key"].append(resp_key)
            trial_data["resp"].append(resp)
            trial_data["fb"].append(fb)
            trial_data["rt"].append(rt)
            trial_data["ts_iso"].append(datetime.now().isoformat())
            trial_data["tms_trigger_enabled"].append(int(bool(TMS_TRIGGER_ENABLED)))
            trial_data["trigger_stim"].append(trig_stim)
            trial_data["trigger_resp"].append(trig_resp)
            trial_data["trigger_fb"].append(trig_fb)
            trial_data["t_stim"].append(flip_times["t_stim"])
            trial_data["t_trial_onset"].append(flip_times["t_trial_onset"])
            trial_data["trigger_pulse_1"].append(pulse_1_code)
            trial_data["trigger_pulse_2"].append(pulse_2_code)
            trial_data["t_pulse_1"].append(flip_times["t_pulse_1"])
            trial_data["t_pulse_2"].append(flip_times["t_pulse_2"])
            trial_data["t_resp"].append(t_resp)
            trial_data["t_fb"].append(flip_times["t_fb"])
            trial_data["port_address"].append(
                TMS_TRIGGER_PORT_ADDRESS if TMS_TRIGGER_ENABLED else "")
            trial_data["block_order"].append(block_order)
            trial_data["x"].append(current_trial["x"])
            trial_data["y"].append(current_trial["y"])
            trial_data["xt"].append(xt)
            trial_data["yt"].append(yt)

            pd.DataFrame(trial_data).to_csv(full_path, index=False)

            next_trial = trial + 1
            if next_trial >= n_total:
                state_current = "state_finished"
            elif (trials["block_num"].iloc[next_trial] !=
                    current_trial["block_num"]):
                state_current = "state_block_start"
            elif (current_trial["block_kind"] in {"rest", "task"} and
                    next_trial % 25 == 0):
                state_current = "state_break"
            else:
                state_current = "state_iti"
            state_entry = True

            win.flip()

    # --------------------------- Cleanup ------------------------------------------
    trigger_port.close()
    win.close()
    core.quit()
    sys.exit()
