from __future__ import annotations

from typing import Any

import numpy as np

C0 = 299792458.0
ZETA0 = 120.0 * np.pi


def _to_array(x: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _blkdiag(*arrays: np.ndarray) -> np.ndarray:
    non_empty = [a for a in arrays if a.size > 0]
    if not non_empty:
        return np.zeros((0, 0), dtype=complex)
    rows = sum(a.shape[0] for a in non_empty)
    cols = sum(a.shape[1] for a in non_empty)
    out = np.zeros((rows, cols), dtype=complex)
    r = 0
    c = 0
    for a in non_empty:
        rr, cc = a.shape
        out[r : r + rr, c : c + cc] = a
        r += rr
        c += cc
    return out


def _inv_or_pinv(a: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a)


def OneModeEigens(a: float, b: float, m: int, n: int) -> tuple[float, float]:
    return m * np.pi / a, n * np.pi / b


def OneModeNormCoeff(a: float, b: float, m: int, n: int, kx: float, ky: float) -> float:
    dm = 1 if m == 0 else 0
    dn = 1 if n == 0 else 0
    return 2.0 / (np.sqrt(a * b) * np.sqrt(kx**2 * (1 + dn) + ky**2 * (1 + dm)))


def OrderModes(segment: dict[str, Any], symmetry: dict[str, int]) -> tuple[dict[str, Any], dict[str, Any]]:
    if segment["a"] > segment["b"]:
        mstart = 1
        nstart = 0
    else:
        mstart = 0
        nstart = 1

    mstep = 1
    mstop = segment["Nmodes"] + 1
    nstep = 1
    nstop = segment["Nmodes"] + 1

    if symmetry.get("x", 0) == 1:
        mstep = 2
    if symmetry.get("y", 0) == 1:
        nstep = 2
    if symmetry.get("H", 0) == 1:
        mstop = mstart
        nstop = nstop * 2
    if symmetry.get("E", 0) == 1:
        nstop = nstart
        mstop = mstop * 2

    tmp: list[tuple[float, int, int]] = []
    for m in range(mstart, mstop + 1, mstep):
        for n in range(nstart, nstop + 1, nstep):
            kx, ky = OneModeEigens(segment["a"], segment["b"], m, n)
            tmp.append((kx**2 + ky**2, m, n))

    tmp.sort(key=lambda t: t[0])
    mh: list[int] = []
    nh: list[int] = []
    me: list[int] = []
    ne: list[int] = []

    im = 0
    while (len(mh) + len(me)) < segment["Nmodes"] and im < len(tmp):
        _, m, n = tmp[im]
        mh.append(m)
        nh.append(n)
        if m > 0 and n > 0:
            me.append(m)
            ne.append(n)
        im += 1

    segment["Nh"] = len(mh)
    segment["Ne"] = len(me)
    segment["mh"] = mh
    segment["nh"] = nh
    segment["me"] = me
    segment["ne"] = ne
    return segment, {"info": [f"modes used: {segment['Nh'] + segment['Ne']}"]}


def EigenModes(segment: dict[str, Any]) -> dict[str, Any]:
    khx = []
    khy = []
    for m, n in zip(segment["mh"], segment["nh"]):
        kx, ky = OneModeEigens(segment["a"], segment["b"], int(m), int(n))
        khx.append(kx)
        khy.append(ky)

    kex = []
    key = []
    for m, n in zip(segment["me"], segment["ne"]):
        kx, ky = OneModeEigens(segment["a"], segment["b"], int(m), int(n))
        kex.append(kx)
        key.append(ky)

    segment["kh"] = {"x": _to_array(khx), "y": _to_array(khy)}
    segment["ke"] = {"x": _to_array(kex), "y": _to_array(key)}
    return segment


def NormCoeff(segment: dict[str, Any]) -> dict[str, Any]:
    ah = []
    for m, n, kx, ky in zip(segment["mh"], segment["nh"], segment["kh"]["x"], segment["kh"]["y"]):
        ah.append(OneModeNormCoeff(segment["a"], segment["b"], int(m), int(n), float(kx), float(ky)))

    ae = []
    for m, n, kx, ky in zip(segment["me"], segment["ne"], segment["ke"]["x"], segment["ke"]["y"]):
        ae.append(OneModeNormCoeff(segment["a"], segment["b"], int(m), int(n), float(kx), float(ky)))

    segment["Ah"] = _to_array(ah)
    segment["Ae"] = _to_array(ae)
    return segment


def WaveNumbers(segment: dict[str, Any], k0: float) -> dict[str, Any]:
    kh = []
    for m, n in zip(segment["mh"], segment["nh"]):
        kmn = -1j * np.sqrt((m * np.pi / segment["a"]) ** 2 + (n * np.pi / segment["b"]) ** 2 - k0**2 + 0j)
        kh.append(kmn)

    ke = []
    for m, n in zip(segment["me"], segment["ne"]):
        kmn = -1j * np.sqrt((m * np.pi / segment["a"]) ** 2 + (n * np.pi / segment["b"]) ** 2 - k0**2 + 0j)
        ke.append(kmn)

    segment["kh"]["mn"] = np.asarray(kh, dtype=complex)
    segment["ke"]["mn"] = np.asarray(ke, dtype=complex)
    return segment


def DelayMatrix(segment: dict[str, Any], k0: float) -> dict[str, Any]:
    del k0
    nh = int(segment["Nh"])
    ne = int(segment["Ne"])
    d11 = np.diag(np.exp(-1j * segment["kh"]["mn"] * segment["l"])) if nh > 0 else np.zeros((0, 0), dtype=complex)
    d22 = np.diag(np.exp(-1j * segment["ke"]["mn"] * segment["l"])) if ne > 0 else np.zeros((0, 0), dtype=complex)
    segment["D"] = np.block(
        [
            [d11, np.zeros((nh, ne), dtype=complex)],
            [np.zeros((ne, nh), dtype=complex), d22],
        ]
    )
    return segment


def Integrals(kx1: float, ky1: float, kx2: float, ky2: float, step: list[dict[str, Any]]) -> np.ndarray:
    a1 = step[0]["a"]
    b1 = step[0]["b"]
    c = abs((step[0]["xo"] - step[0]["a"] / 2.0) - (step[1]["xo"] - step[1]["a"] / 2.0))
    d = abs((step[0]["yo"] - step[0]["b"] / 2.0) - (step[1]["yo"] - step[1]["b"] / 2.0))

    def term(p: float, half_len: float, phase: float) -> float:
        x = p * half_len
        if abs(x) > np.finfo(float).eps:
            return half_len * np.sin(x) / x * np.cos(x + phase)
        return half_len * np.cos(x + phase)

    i1 = term((ky1 - ky2), 0.5 * b1, -ky2 * d)
    i2 = term((ky1 + ky2), 0.5 * b1, ky2 * d)
    i3 = term((kx1 - kx2), 0.5 * a1, -kx2 * c)
    i4 = term((kx1 + kx2), 0.5 * a1, kx2 * c)
    return np.asarray([i1, i2, i3, i4], dtype=float)


def MxxMatrices(step: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s1 = step[0]
    s2 = step[1]
    mhh = np.zeros((s1["Nh"], s2["Nh"]), dtype=complex)
    for i in range(s1["Nh"]):
        for j in range(s2["Nh"]):
            iint = Integrals(s1["kh"]["x"][i], s1["kh"]["y"][i], s2["kh"]["x"][j], s2["kh"]["y"][j], step)
            mhh[i, j] = s1["Ah"][i] * s2["Ah"][j] * (
                s1["kh"]["y"][i] * s2["kh"]["y"][j] * (iint[0] - iint[1]) * (iint[2] + iint[3])
                + s1["kh"]["x"][i] * s2["kh"]["x"][j] * (iint[0] + iint[1]) * (iint[2] - iint[3])
            )

    mhe = np.zeros((s1["Nh"], s2["Ne"]), dtype=complex)
    for i in range(s1["Nh"]):
        for j in range(s2["Ne"]):
            iint = Integrals(s1["kh"]["x"][i], s1["kh"]["y"][i], s2["ke"]["x"][j], s2["ke"]["y"][j], step)
            mhe[i, j] = s1["Ah"][i] * s2["Ae"][j] * (
                -s1["kh"]["y"][i] * s2["ke"]["x"][j] * (iint[0] - iint[1]) * (iint[2] + iint[3])
                + s1["kh"]["x"][i] * s2["ke"]["y"][j] * (iint[0] + iint[1]) * (iint[2] - iint[3])
            )

    meh = np.zeros((s1["Ne"], s2["Nh"]), dtype=complex)
    mee = np.zeros((s1["Ne"], s2["Ne"]), dtype=complex)
    for i in range(s1["Ne"]):
        for j in range(s2["Ne"]):
            iint = Integrals(s1["ke"]["x"][i], s1["ke"]["y"][i], s2["ke"]["x"][j], s2["ke"]["y"][j], step)
            mee[i, j] = s1["Ae"][i] * s2["Ae"][j] * (
                s1["ke"]["x"][i] * s2["ke"]["x"][j] * (iint[0] - iint[1]) * (iint[2] + iint[3])
                + s1["ke"]["y"][i] * s2["ke"]["y"][j] * (iint[0] + iint[1]) * (iint[2] - iint[3])
            )

    return mhh, mhe, meh, mee


def DgammaMatrices(step: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        1j * np.diag(step[0]["kh"]["mn"]),
        1j * np.diag(step[1]["kh"]["mn"]),
        1j * np.diag(step[0]["ke"]["mn"]),
        1j * np.diag(step[1]["ke"]["mn"]),
    )


def UMatrices(step: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.eye(step[0]["Nh"], dtype=complex),
        np.eye(step[1]["Nh"], dtype=complex),
        np.eye(step[0]["Ne"], dtype=complex),
        np.eye(step[1]["Ne"], dtype=complex),
    )


def SingleStep(wgs: list[dict[str, Any]], segment: int, k0: float) -> dict[str, np.ndarray]:
    left = wgs[segment]
    right = wgs[segment + 1]
    if left["a"] <= right["a"] and left["b"] <= right["b"]:
        step = [left, right]
        rflag = 0
    else:
        step = [right, left]
        rflag = 1

    mhh, mhe, meh, mee = MxxMatrices(step)
    dg_h_1, dg_h_2, dg_e_1, dg_e_2 = DgammaMatrices(step)
    uh_1, uh_2, ue_1, _ue_2 = UMatrices(step)

    lamb = 2 * np.pi / k0
    z_l = 1j * 2 * np.pi * ZETA0 / lamb
    y_l = 1j * 2 * np.pi / (ZETA0 * lamb)

    dh = np.block(
        [
            [dg_h_1, np.zeros((step[0]["Nh"], step[0]["Ne"]), dtype=complex)],
            [np.zeros((step[0]["Ne"], step[0]["Nh"]), dtype=complex), ue_1],
        ]
    )
    mh = np.block(
        [
            [mhh @ dg_h_2, y_l * mhe],
            [meh @ dg_h_2 / y_l if step[0]["Ne"] > 0 else meh, mee],
        ]
    )

    de = np.block(
        [
            [uh_2, np.zeros((step[1]["Nh"], step[1]["Ne"]), dtype=complex)],
            [np.zeros((step[1]["Ne"], step[1]["Nh"]), dtype=complex), dg_e_2],
        ]
    )
    me = np.block(
        [
            [mhh.T, meh.T @ dg_e_1 / z_l if step[1]["Nh"] > 0 else meh.T],
            [z_l * mhe.T, mee.T @ dg_e_1 if step[1]["Ne"] > 0 else mee.T],
        ]
    )

    h = _inv_or_pinv(dh) @ mh
    e = _inv_or_pinv(de) @ me
    eye1 = np.eye(step[0]["Nh"] + step[0]["Ne"], dtype=complex)
    s11 = _inv_or_pinv(eye1 + h @ e) @ (eye1 - h @ e)
    s12 = 2 * (_inv_or_pinv(eye1 + h @ e) @ h)
    s21 = e @ (eye1 + s11)
    s22 = e @ s12 - np.eye(step[1]["Nh"] + step[1]["Ne"], dtype=complex)

    if rflag == 0:
        return {"S11": s11, "S12": s12, "S21": s21, "S22": s22}
    return {"S11": s22, "S12": s21, "S21": s12, "S22": s11}


def SingleCascade(sa: dict[str, np.ndarray], d: np.ndarray, sb: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    p = _inv_or_pinv(np.eye(d.shape[0], dtype=complex) - d @ sb["S11"] @ d @ sa["S22"])
    q = _inv_or_pinv(np.eye(d.shape[0], dtype=complex) - d @ sa["S22"] @ d @ sb["S11"])

    left = np.block(
        [
            [sa["S12"], np.zeros((sa["S12"].shape[0], sb["S21"].shape[1]), dtype=complex)],
            [np.zeros((sb["S21"].shape[0], sa["S12"].shape[1]), dtype=complex), sb["S21"]],
        ]
    )
    mid = np.block([[p @ d @ sb["S11"], p], [q, q @ d @ sa["S22"]]])
    right = np.block(
        [
            [d @ sa["S21"], np.zeros((sa["S21"].shape[0], sb["S12"].shape[1]), dtype=complex)],
            [np.zeros((sb["S12"].shape[0], sa["S21"].shape[1]), dtype=complex), d @ sb["S12"]],
        ]
    )
    add = np.block(
        [
            [sa["S11"], np.zeros((sa["S11"].shape[0], sb["S22"].shape[1]), dtype=complex)],
            [np.zeros((sb["S22"].shape[0], sa["S11"].shape[1]), dtype=complex), sb["S22"]],
        ]
    )
    sg = left @ mid @ right + add

    na, ma = sa["S11"].shape
    nb, mb = sb["S22"].shape
    return {
        "S11": sg[:na, :ma],
        "S12": sg[:na, ma : ma + mb],
        "S21": sg[na : na + nb, :ma],
        "S22": sg[na : na + nb, ma : ma + mb],
    }


def Cascade(wgs: list[dict[str, Any]], s_steps: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    n0 = wgs[0]["Nh"] + wgs[0]["Ne"]
    s0 = {"S11": np.zeros((n0, n0), dtype=complex), "S12": np.eye(n0, dtype=complex), "S21": np.eye(n0, dtype=complex), "S22": np.zeros((n0, n0), dtype=complex)}
    stot = SingleCascade(s0, wgs[0]["D"], s_steps[0])

    nparts = len(wgs)
    if nparts > 1:
        for p in range(1, nparts - 1):
            stot = SingleCascade(stot, wgs[p]["D"], s_steps[p])

        nlast = wgs[-1]["Nh"] + wgs[-1]["Ne"]
        slast = {"S11": np.zeros((nlast, nlast), dtype=complex), "S12": np.eye(nlast, dtype=complex), "S21": np.eye(nlast, dtype=complex), "S22": np.zeros((nlast, nlast), dtype=complex)}
        stot = SingleCascade(stot, wgs[-1]["D"], slast)

    return stot


def MultiStep(wgs: list[dict[str, Any]], k0: float, symmetry: dict[str, int]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    del symmetry
    for p in range(len(wgs)):
        WaveNumbers(wgs[p], k0)
        DelayMatrix(wgs[p], k0)

    if len(wgs) > 1:
        s_steps = [SingleStep(wgs, p, k0) for p in range(len(wgs) - 1)]
    else:
        n = wgs[0]["Nh"] + wgs[0]["Ne"]
        s_steps = [{"S11": np.zeros((n, n), dtype=complex), "S12": np.eye(n, dtype=complex), "S21": np.eye(n, dtype=complex), "S22": np.zeros((n, n), dtype=complex)}]

    stot = Cascade(wgs, s_steps)
    sf = np.block([[stot["S11"], stot["S12"]], [stot["S21"], stot["S22"]]])
    return sf, wgs


def Nto1Junction(side_one: list[dict[str, Any]], side_two: dict[str, Any], k0: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n_segments = len(side_one)
    h_blocks = []
    e_blocks = []
    lamb = 2 * np.pi / k0
    z_l = 1j * 2 * np.pi * ZETA0 / lamb
    y_l = 1j * 2 * np.pi / (ZETA0 * lamb)

    total_in_modes = 0
    for i in range(n_segments):
        step = [side_one[i], side_two]
        mhh, mhe, meh, mee = MxxMatrices(step)
        dg_h_1, dg_h_2, dg_e_1, dg_e_2 = DgammaMatrices(step)

        dh = np.block(
            [
                [dg_h_1, np.zeros((step[0]["Nh"], step[0]["Ne"]), dtype=complex)],
                [np.zeros((step[0]["Ne"], step[0]["Nh"]), dtype=complex), np.eye(step[0]["Ne"], dtype=complex)],
            ]
        )
        mh = np.block(
            [
                [mhh @ dg_h_2, y_l * mhe],
                [meh @ dg_h_2 / y_l if step[0]["Ne"] > 0 else meh, mee],
            ]
        )
        de = np.block(
            [
                [np.eye(step[1]["Nh"], dtype=complex), np.zeros((step[1]["Nh"], step[1]["Ne"]), dtype=complex)],
                [np.zeros((step[1]["Ne"], step[1]["Nh"]), dtype=complex), dg_e_2],
            ]
        )
        me = np.block(
            [
                [mhh.T, meh.T @ dg_e_1 / z_l if step[1]["Nh"] > 0 else meh.T],
                [z_l * mhe.T, mee.T @ dg_e_1 if step[1]["Ne"] > 0 else mee.T],
            ]
        )

        h_blocks.append(_inv_or_pinv(dh) @ mh)
        e_blocks.append(_inv_or_pinv(de) @ me)
        total_in_modes += step[0]["Nh"] + step[0]["Ne"]

    h = np.vstack(h_blocks)
    e = np.hstack(e_blocks)
    he = h @ e
    u_in = np.eye(total_in_modes, dtype=complex)
    u_out = np.eye(side_two["Nh"] + side_two["Ne"], dtype=complex)
    inv_u = _inv_or_pinv(u_in + he)
    s11 = inv_u @ (u_in - he)
    s12 = 2 * inv_u @ h
    s21 = 2 * e @ inv_u
    s22 = e @ s12 - u_out

    s = {"S11": s11, "S12": s12, "S21": s21, "S22": s22}
    stot = np.block([[s11, s12], [s21, s22]])
    return stot, s


def ExtractPortS(stot: np.ndarray, sele_dim: tuple[int, int], i: int, j: int) -> np.ndarray:
    r0 = (i - 1) * sele_dim[0]
    c0 = (j - 1) * sele_dim[1]
    return stot[r0 : r0 + sele_dim[0], c0 : c0 + sele_dim[1]]


def InsertPortS(stot: np.ndarray, s: np.ndarray, sele_dim: tuple[int, int], i: int, j: int) -> np.ndarray:
    if stot.size == 0:
        n = max(i, j)
        stot = np.zeros((n * sele_dim[0], n * sele_dim[1]), dtype=complex)
    r0 = (i - 1) * sele_dim[0]
    c0 = (j - 1) * sele_dim[1]
    rr = max(stot.shape[0], r0 + sele_dim[0])
    cc = max(stot.shape[1], c0 + sele_dim[1])
    if rr != stot.shape[0] or cc != stot.shape[1]:
        grown = np.zeros((rr, cc), dtype=complex)
        grown[: stot.shape[0], : stot.shape[1]] = stot
        stot = grown
    block = np.zeros((sele_dim[0], sele_dim[1]), dtype=complex)
    block[: min(sele_dim[0], s.shape[0]), : min(sele_dim[1], s.shape[1])] = s[: min(sele_dim[0], s.shape[0]), : min(sele_dim[1], s.shape[1])]
    stot[r0 : r0 + sele_dim[0], c0 : c0 + sele_dim[1]] = block
    return stot


def CondenseGSM(s: np.ndarray, sele_dim: tuple[int, int], topology: dict[str, Any]) -> np.ndarray:
    work = s.copy()
    nbr_ports = work.shape[0] // sele_dim[0]
    index_map = {idx: idx for idx in range(1, nbr_ports + 1)}

    for orig_m, orig_n in zip(topology["PortToCondense"]["a"], topology["PortToCondense"]["b"]):
        if orig_m not in index_map or orig_n not in index_map:
            continue
        indexm = index_map[orig_m]
        indexn = index_map[orig_n]
        if indexm == indexn:
            continue
        if indexm > indexn:
            indexm, indexn = indexn, indexm

        c = ExtractPortS(work, sele_dim, indexm, indexm)
        d = ExtractPortS(work, sele_dim, indexm, indexn) - np.eye(sele_dim[0], dtype=complex)
        e = ExtractPortS(work, sele_dim, indexn, indexm) - np.eye(sele_dim[0], dtype=complex)
        f = ExtractPortS(work, sele_dim, indexn, indexn)

        if np.linalg.matrix_rank(c) == c.shape[0]:
            xi = e @ _inv_or_pinv(c)
            phi_nm = _inv_or_pinv(xi @ d - f) @ xi
            phi_nn = -_inv_or_pinv(xi @ d - f)
        else:
            xi = c @ _inv_or_pinv(e)
            phi_nm = _inv_or_pinv(d - xi @ f)
            phi_nn = -_inv_or_pinv(d - xi @ f) @ xi

        if np.linalg.matrix_rank(f) == f.shape[0]:
            psi = d @ _inv_or_pinv(f)
            phi_mm = -_inv_or_pinv(psi @ e - c)
            phi_mn = _inv_or_pinv(psi @ e - c) @ psi
        else:
            psi = f @ _inv_or_pinv(d)
            phi_mm = -_inv_or_pinv(e - psi @ c) @ psi
            phi_mn = _inv_or_pinv(e - psi @ c)

        keep_ports = [p for p in range(1, nbr_ports + 1) if p not in (indexm, indexn)]
        condensed = np.zeros((len(keep_ports) * sele_dim[0], len(keep_ports) * sele_dim[1]), dtype=complex)

        for oi, i in enumerate(keep_ports, start=1):
            for oj, j in enumerate(keep_ports, start=1):
                stmp = (
                    ExtractPortS(work, sele_dim, i, j)
                    - ExtractPortS(work, sele_dim, i, indexm)
                    @ (phi_mm @ ExtractPortS(work, sele_dim, indexm, j) + phi_mn @ ExtractPortS(work, sele_dim, indexn, j))
                    - ExtractPortS(work, sele_dim, i, indexn)
                    @ (phi_nm @ ExtractPortS(work, sele_dim, indexm, j) + phi_nn @ ExtractPortS(work, sele_dim, indexn, j))
                )
                condensed = InsertPortS(condensed, stmp, sele_dim, oi, oj)

        inv_keep_map = {old: new for new, old in enumerate(keep_ports, start=1)}
        index_map = {old_port: inv_keep_map[cur_idx] for old_port, cur_idx in index_map.items() if cur_idx in inv_keep_map}
        work = condensed
        nbr_ports = len(keep_ports)

    sout = np.zeros((0, 0), dtype=complex)
    for i_out, orig_i in enumerate(topology["OpenPorts"], start=1):
        for j_out, orig_j in enumerate(topology["OpenPorts"], start=1):
            i = index_map[orig_i]
            j = index_map[orig_j]
            sout = InsertPortS(sout, ExtractPortS(work, sele_dim, i, j), sele_dim, i_out, j_out)
    return sout


def Renormalize(wgs: list[dict[str, Any]], stot: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    w1 = wgs[0]
    w2 = wgs[-1]
    nh1, ne1 = w1["Nh"], w1["Ne"]
    nh2, ne2 = w2["Nh"], w2["Ne"]

    out = {
        "S11": np.zeros((nh1 + ne1, nh1 + ne1), dtype=complex),
        "S12": np.zeros((nh1 + ne1, nh2 + ne2), dtype=complex),
        "S21": np.zeros((nh2 + ne2, nh1 + ne1), dtype=complex),
        "S22": np.zeros((nh2 + ne2, nh2 + ne2), dtype=complex),
    }

    kh1 = w1["kh"]["mn"]
    ke1 = w1["ke"]["mn"]
    kh2 = w2["kh"]["mn"]
    ke2 = w2["ke"]["mn"]

    for i in range(nh1):
        for j in range(nh1):
            out["S11"][i, j] = stot["S11"][i, j] * np.sqrt(kh1[i] / kh1[j])
    for i in range(nh1):
        for j in range(ne1):
            out["S11"][i, nh1 + j] = stot["S11"][i, nh1 + j] * np.sqrt(kh1[i] / ke1[j]) * ZETA0
    for i in range(ne1):
        for j in range(nh1):
            out["S11"][nh1 + i, j] = stot["S11"][nh1 + i, j] * np.sqrt(ke1[i] / kh1[j]) / ZETA0
    for i in range(ne1):
        for j in range(ne1):
            out["S11"][nh1 + i, nh1 + j] = stot["S11"][nh1 + i, nh1 + j] * np.sqrt(ke1[i] / ke1[j])

    for i in range(nh2):
        for j in range(nh1):
            out["S21"][i, j] = stot["S21"][i, j] * np.sqrt(kh2[i] / kh1[j])
    for i in range(nh2):
        for j in range(ne1):
            out["S21"][i, nh1 + j] = stot["S21"][i, nh1 + j] * np.sqrt(kh2[i] / ke1[j]) * ZETA0
    for i in range(ne2):
        for j in range(nh1):
            out["S21"][nh2 + i, j] = stot["S21"][nh2 + i, j] * np.sqrt(ke2[i] / kh1[j]) / ZETA0
    for i in range(ne2):
        for j in range(ne1):
            out["S21"][nh2 + i, nh1 + j] = stot["S21"][nh2 + i, nh1 + j] * np.sqrt(ke2[i] / ke1[j])

    for i in range(nh1):
        for j in range(nh2):
            out["S12"][i, j] = stot["S12"][i, j] * np.sqrt(kh1[i] / kh2[j])
    for i in range(nh1):
        for j in range(ne2):
            out["S12"][i, nh2 + j] = stot["S12"][i, nh2 + j] * np.sqrt(kh1[i] / ke2[j]) * ZETA0
    for i in range(ne1):
        for j in range(nh2):
            out["S12"][nh1 + i, j] = stot["S12"][nh1 + i, j] * np.sqrt(ke1[i] / kh2[j]) / ZETA0
    for i in range(ne1):
        for j in range(ne2):
            out["S12"][nh1 + i, nh2 + j] = stot["S12"][nh1 + i, nh2 + j] * np.sqrt(ke1[i] / ke2[j])

    for i in range(nh2):
        for j in range(nh2):
            out["S22"][i, j] = stot["S22"][i, j] * np.sqrt(kh2[i] / kh2[j])
    for i in range(nh2):
        for j in range(ne2):
            out["S22"][i, nh2 + j] = stot["S22"][i, nh2 + j] * np.sqrt(kh2[i] / ke2[j]) * ZETA0
    for i in range(ne2):
        for j in range(nh2):
            out["S22"][nh2 + i, j] = stot["S22"][nh2 + i, j] * np.sqrt(ke2[i] / kh2[j]) / ZETA0
    for i in range(ne2):
        for j in range(ne2):
            out["S22"][nh2 + i, nh2 + j] = stot["S22"][nh2 + i, nh2 + j] * np.sqrt(ke2[i] / ke2[j])

    return out


def RenormalizeGSM(s: np.ndarray, sele_dim: tuple[int, int], side_one: list[dict[str, Any]], side_two: list[dict[str, Any]]) -> np.ndarray:
    sdim = s.shape[0] // sele_dim[0]
    work = s.copy()

    for i in range(len(side_one)):
        for j in range(i + 1, len(side_one)):
            stmp = {
                "S11": ExtractPortS(work, sele_dim, i + 1, i + 1),
                "S12": ExtractPortS(work, sele_dim, i + 1, j + 1),
                "S21": ExtractPortS(work, sele_dim, j + 1, i + 1),
                "S22": ExtractPortS(work, sele_dim, j + 1, j + 1),
            }
            stmp_rn = Renormalize([side_one[i], side_one[j]], stmp)
            work = InsertPortS(work, stmp_rn["S11"], sele_dim, i + 1, i + 1)
            work = InsertPortS(work, stmp_rn["S12"], sele_dim, i + 1, j + 1)
            work = InsertPortS(work, stmp_rn["S21"], sele_dim, j + 1, i + 1)
            work = InsertPortS(work, stmp_rn["S22"], sele_dim, j + 1, j + 1)

    for w2 in side_two:
        for i in range(len(side_one)):
            stmp = {
                "S11": ExtractPortS(work, sele_dim, i + 1, i + 1),
                "S12": ExtractPortS(work, sele_dim, i + 1, sdim),
                "S21": ExtractPortS(work, sele_dim, sdim, i + 1),
                "S22": ExtractPortS(work, sele_dim, sdim, sdim),
            }
            stmp_rn = Renormalize([side_one[i], w2], stmp)
            work = InsertPortS(work, stmp_rn["S11"], sele_dim, i + 1, i + 1)
            work = InsertPortS(work, stmp_rn["S12"], sele_dim, i + 1, sdim)
            work = InsertPortS(work, stmp_rn["S21"], sele_dim, sdim, i + 1)
            work = InsertPortS(work, stmp_rn["S22"], sele_dim, sdim, sdim)

    return work


def ReverseWaveGuideStructure(wgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(reversed(wgs))


def TwoPortDeviceGetPortSegment(wgs: list[dict[str, Any]], port_related: int) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if port_related == 1:
        return wgs[0], wgs[-1], wgs
    if port_related == 2:
        return wgs[-1], wgs[0], wgs
    raise ValueError("PortRelated must be 1 or 2")


def FrequencySweepValidate(fs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    out = dict(fs)
    if "f" in out:
        f = np.asarray(out["f"], dtype=float)
        if np.any(f < np.finfo(float).eps):
            return out, {"fatal": "FrequencySweep.f contains non-positive entries"}
        out["f"] = f
        out["N"] = int(f.size)
        return out, {}

    n = int(out["N"])
    fstart = float(out["start"])
    fend = float(out["end"])
    out["f"] = np.linspace(fstart, fend, n) if n >= 2 else np.asarray([fstart], dtype=float)
    out["N"] = int(out["f"].size)
    return out, {}


def MultiPortDeviceTopology(two_port_devices: list[dict[str, Any]], nto1_connections: list[dict[str, Any]], open_ports: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    del two_port_devices
    del open_ports
    top: dict[str, Any] = {"Nto1": []}
    for conn in nto1_connections:
        side_one_count = len(conn["SideOne"])
        side_two_count = len(conn["SideTwo"])
        dims = side_one_count + side_two_count
        idx_for_nto1 = 2 * dims
        t = {"Dimensions": dims, "PortToCondense": {"a": [], "b": []}, "OpenPorts": []}

        if side_one_count > 1 and side_two_count == 1:
            t["nFurcation"] = 0
            for i in range(1, side_one_count + 1):
                t["PortToCondense"]["a"].append(2 * i)
                t["PortToCondense"]["b"].append(idx_for_nto1 + i)
                t["OpenPorts"].append(2 * i - 1)
            for i in range(1, side_two_count + 1):
                t["PortToCondense"]["a"].append(idx_for_nto1 + side_one_count + i)
                t["PortToCondense"]["b"].append(side_one_count * 2 + i * 2 - 1)
                t["OpenPorts"].append(2 * dims)
        elif side_two_count > 1 and side_one_count == 1:
            t["nFurcation"] = 1
            for i in range(1, side_two_count + 1):
                t["PortToCondense"]["a"].append(2 * i)
                t["PortToCondense"]["b"].append(idx_for_nto1 + i)
                t["OpenPorts"].append(2 * i - 1)
            for i in range(1, side_one_count + 1):
                t["PortToCondense"]["a"].append(idx_for_nto1 + side_two_count + i)
                t["PortToCondense"]["b"].append(side_two_count * 2 + i * 2 - 1)
                t["OpenPorts"].append(2 * dims)
        else:
            return {}, {"fatal": "Nto1 Connection has not been defined correctly"}

        top["Nto1"].append(t)

    return top, {}


def MultiPortDeviceValidate(
    two_port_devices: list[dict[str, Any]],
    nto1_connections: list[dict[str, Any]],
    open_ports: list[dict[str, Any]],
    connected_ports: list[Any],
    frequency_sweep: dict[str, Any],
    flag: int,
    options: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[Any], dict[str, Any], dict[str, int], dict[str, Any], dict[str, Any]]:
    # Display device if flag > 0
    if flag:
        MultiPortDeviceDraw(two_port_devices, nto1_connections, max(0, flag - 1), options)
    
    fs, err = FrequencySweepValidate(frequency_sweep)
    if err.get("fatal"):
        return two_port_devices, nto1_connections, open_ports, connected_ports, fs, {}, {}, err

    symmetry = {"x": 0, "y": 0, "H": 0, "E": 0}
    top, terr = MultiPortDeviceTopology(two_port_devices, nto1_connections, open_ports)
    if terr.get("fatal"):
        return two_port_devices, nto1_connections, open_ports, connected_ports, fs, symmetry, {}, terr

    return two_port_devices, nto1_connections, open_ports, connected_ports, fs, symmetry, top, {}


def _stmp_port_to_sinfo_idx(local_port: int, n_side_one: int, n_side_two: int, n_furcation: int) -> int:
    """Map a condensed-Stmp local port (1-indexed) to the index in conn_sinfo.

    conn_sinfo ordering: [SideOne_0, ..., SideOne_N-1, SideTwo_0, ..., SideTwo_M-1]
    For nFurcation==0 (N→1): stmp port i → conn_sinfo[i-1]  (SideOne first, then SideTwo)
    For nFurcation==1 (1→N): FutureGSM=blkdiag(SideTwo_rev, SideOne_rev), so
        stmp port 1..N_two → SideTwo devices → conn_sinfo[n_side_one + (local_port-1)]
        stmp port N_two+1  → SideOne device  → conn_sinfo[0]
    """
    if n_furcation == 0:
        return local_port - 1
    else:
        if local_port <= n_side_two:
            return n_side_one + (local_port - 1)
        return 0


def MultiPortDeviceSolve(
    two_port_devices: list[dict[str, Any]],
    nto1_connections: list[dict[str, Any]],
    open_ports: list[dict[str, Any]],
    connected_ports: list[Any],
    frequency_sweep: dict[str, Any],
    symmetry: dict[str, int],
    topology: dict[str, Any],
    options: dict[str, Any],
) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[Any]]:
    # Preprocess ConnectedPorts: cascade directly-connected device pairs by merging their D-lists
    # and remapping all Nto1 and OpenPorts references from the consumed device to the host device.
    for cp in connected_ports:
        raw_idx = cp.get("TwoPortDeviceIndex", [])
        raw_port = cp.get("TwoPortDevicePort", [])
        if len(raw_idx) != 2 or len(raw_port) != 2:
            continue
        a_idx, b_idx = int(raw_idx[0]), int(raw_idx[1])
        a_port, b_port = int(raw_port[0]), int(raw_port[1])
        if a_port == 2 and b_port == 1:
            merged_d = list(two_port_devices[a_idx - 1]["D"]) + list(two_port_devices[b_idx - 1]["D"])
        elif a_port == 1 and b_port == 2:
            merged_d = list(two_port_devices[b_idx - 1]["D"]) + list(two_port_devices[a_idx - 1]["D"])
        else:
            continue
        two_port_devices[a_idx - 1]["D"] = merged_d
        two_port_devices[b_idx - 1]["D"] = []
        for conn in nto1_connections:
            for s in conn.get("SideOne", []):
                if int(s["TwoPortDeviceIndex"]) == b_idx:
                    s["TwoPortDeviceIndex"] = a_idx
            for s in conn.get("SideTwo", []):
                if int(s["TwoPortDeviceIndex"]) == b_idx:
                    s["TwoPortDeviceIndex"] = a_idx
        for p in open_ports:
            if int(p["TwoPortDeviceIndex"]) == b_idx:
                p["TwoPortDeviceIndex"] = a_idx
    sym_use = options.get("DeviceSymmetry", {}).get("Use", 0)
    sym_side = options.get("DeviceSymmetry", {}).get("Side", 2)

    for dev in two_port_devices:
        for seg in dev["D"]:
            seg, _ = OrderModes(seg, symmetry)
            EigenModes(seg)
            NormCoeff(seg)

    open_dev_indices = set(int(p["TwoPortDeviceIndex"]) for p in open_ports)
    n_conns = len(nto1_connections)

    # Build LocalPortstoMerge: for each pair of consecutive Nto1s that share a device,
    # record the (conn_idx, local_port_1indexed) for both sides.
    local_ports_to_merge: dict[str, list[Any]] = {"a": [], "b": []}
    for ci in range(1, n_conns):
        cur_conn = nto1_connections[ci]
        cur_t = topology["Nto1"][ci]
        for prev_ci in range(ci):
            prev_conn = nto1_connections[prev_ci]
            prev_t = topology["Nto1"][prev_ci]
            for alpha_0, alpha_dev in enumerate(prev_conn["SideTwo"]):
                for beta_0, beta_dev in enumerate(cur_conn["SideOne"]):
                    if alpha_dev["TwoPortDeviceIndex"] == beta_dev["TwoPortDeviceIndex"]:
                        alpha = alpha_0 + 1  # 1-indexed within prev Stmp
                        beta = beta_0 + 1    # 1-indexed within cur Stmp
                        if prev_t["nFurcation"] == 0:
                            alpha += len(prev_conn["SideOne"])
                        if cur_t["nFurcation"] == 1:
                            beta += len(cur_conn["SideTwo"])
                        local_ports_to_merge["a"].append((prev_ci, alpha))
                        local_ports_to_merge["b"].append((ci, beta))

    # Collect the set of device indices that are "shared" (appear in a later Nto1's SideOne
    # after having been processed in an earlier Nto1's SideTwo).
    solved_dev_indices: set[int] = set()
    for ci in range(1, n_conns):
        for prev_ci in range(ci):
            prev_conn = nto1_connections[prev_ci]
            cur_conn = nto1_connections[ci]
            for alpha_dev in prev_conn["SideTwo"]:
                for beta_dev in cur_conn["SideOne"]:
                    if alpha_dev["TwoPortDeviceIndex"] == beta_dev["TwoPortDeviceIndex"]:
                        solved_dev_indices.add(int(beta_dev["TwoPortDeviceIndex"]))

    sf: list[np.ndarray] = []
    sinfo: list[dict[str, Any]] = []
    sele_dim: tuple[int, int] = (0, 0)

    for f_val in frequency_sweep["f"]:
        k0 = 2 * np.pi * f_val / C0
        stmp_list: list[np.ndarray] = []
        conn_sinfos: list[list[dict[str, Any]]] = []
        global_open_side_one: list[dict[str, Any]] = []
        global_open_side_two: list[dict[str, Any]] = []

        for ci, conn in enumerate(nto1_connections):
            t = topology["Nto1"][ci]
            to_rev = int(t["nFurcation"])
            side_one = conn["SideOne"]
            side_two = conn["SideTwo"]
            d1 = [int(s["TwoPortDeviceIndex"]) for s in side_one]
            p1 = [int(s["TwoPortDevicePort"]) for s in side_one]
            d2 = [int(s["TwoPortDeviceIndex"]) for s in side_two]
            p2 = [int(s["TwoPortDevicePort"]) for s in side_two]

            seg_side_one: list[dict[str, Any]] = []
            open_seg_side_one: list[dict[str, Any]] = []
            seg_side_two: list[dict[str, Any]] = []
            open_seg_side_two: list[dict[str, Any]] = []
            conn_sinfo: list[dict[str, Any]] = []

            for i in range(len(d1)):
                seg, opseg, _ = TwoPortDeviceGetPortSegment(two_port_devices[d1[i] - 1]["D"], p1[i])
                seg_side_one.append(seg)
                open_seg_side_one.append(opseg)
                conn_sinfo.append({"mh": opseg["mh"], "nh": opseg["nh"], "me": opseg["me"], "ne": opseg["ne"]})

            for i in range(len(d2)):
                seg, opseg, _ = TwoPortDeviceGetPortSegment(two_port_devices[d2[i] - 1]["D"], p2[i])
                seg_side_two.append(seg)
                open_seg_side_two.append(opseg)
                conn_sinfo.append({"mh": opseg["mh"], "nh": opseg["nh"], "me": opseg["me"], "ne": opseg["ne"]})

            conn_sinfos.append(conn_sinfo)

            # Check uniform mode count
            this_dim = (len(conn_sinfo[0]["mh"]) + len(conn_sinfo[0]["me"]),
                        len(conn_sinfo[0]["nh"]) + len(conn_sinfo[0]["ne"]))
            for si in conn_sinfo[1:]:
                si_dim = (len(si["mh"]) + len(si["me"]), len(si["nh"]) + len(si["ne"]))
                if si_dim != this_dim:
                    return [], [], {"fatal": f"Port mode count mismatch: {this_dim} vs {si_dim}. Adjust Nmodes so all ports share the same total mode count."}, two_port_devices, nto1_connections, []
            sele_dim = this_dim

            # Build FutureGSM
            future_side_one = np.zeros((0, 0), dtype=complex)
            for i in range(len(d1)):
                dev_idx = d1[i]
                if dev_idx in solved_dev_indices:
                    # Device already traversed in a previous Nto1: use zero-length copy
                    wgs_zl = [dict(seg_side_one[i])]
                    wgs_zl[0]["l"] = 0.0
                    s_tp, _ = MultiStep(wgs_zl, k0, symmetry)
                else:
                    wgs = two_port_devices[dev_idx - 1]["D"]
                    if to_rev == 1:
                        wgs = ReverseWaveGuideStructure(wgs)
                    s_tp, updated = MultiStep(wgs, k0, symmetry)
                    two_port_devices[dev_idx - 1]["D"] = ReverseWaveGuideStructure(updated) if to_rev == 1 else updated
                future_side_one = _blkdiag(future_side_one, s_tp)

            future_side_two = np.zeros((0, 0), dtype=complex)
            for i in range(len(d2)):
                dev_idx = d2[i]
                wgs = two_port_devices[dev_idx - 1]["D"]
                if to_rev == 1:
                    wgs = ReverseWaveGuideStructure(wgs)
                s_tp, updated = MultiStep(wgs, k0, symmetry)
                two_port_devices[dev_idx - 1]["D"] = ReverseWaveGuideStructure(updated) if to_rev == 1 else updated
                future_side_two = _blkdiag(future_side_two, s_tp)

            if to_rev == 1:
                future_gsm = _blkdiag(future_side_two, future_side_one)
                s_nto1, _ = Nto1Junction(seg_side_two, seg_side_one[0], k0)
            else:
                future_gsm = _blkdiag(future_side_one, future_side_two)
                s_nto1, _ = Nto1Junction(seg_side_one, seg_side_two[0], k0)

            gsm = _blkdiag(future_gsm, s_nto1)
            stmp_list.append(CondenseGSM(gsm, sele_dim, t))

            # Accumulate open-port segments for renormalization
            for i, dev_idx in enumerate(d1):
                if dev_idx in open_dev_indices:
                    global_open_side_one.append(open_seg_side_one[i])
            for i, dev_idx in enumerate(d2):
                if dev_idx in open_dev_indices:
                    global_open_side_two.append(open_seg_side_two[i])

        # --- Merge multiple stmps if needed ---
        if n_conns == 1:
            sout = stmp_list[0]
            # sinfo: conn_sinfos[0] already in SideOne-first order, matching stmp ports
            sinfo = conn_sinfos[0]
        else:
            new_gsm = np.zeros((0, 0), dtype=complex)
            for s in stmp_list:
                new_gsm = _blkdiag(new_gsm, s)

            new_gsm_size = sum(topology["Nto1"][ci]["Dimensions"] for ci in range(n_conns))

            # Build PortToCondense for the merged matrix
            ptc_a: list[int] = []
            ptc_b: list[int] = []
            for k in range(len(local_ports_to_merge["a"])):
                ci_a, port_a = local_ports_to_merge["a"][k]
                sum_a = sum(topology["Nto1"][j]["Dimensions"] for j in range(ci_a))
                ptc_a.append(sum_a + port_a)
                ci_b, port_b = local_ports_to_merge["b"][k]
                sum_b = sum(topology["Nto1"][j]["Dimensions"] for j in range(ci_b))
                ptc_b.append(sum_b + port_b)

            # Mark condensed ports as unavailable
            remaining = list(range(1, new_gsm_size + 1))
            for p in ptc_a + ptc_b:
                remaining[p - 1] = 0

            # Build final OpenPorts respecting per-Nto1 traversal order
            open_ports_list: list[int] = []
            prev_ports = 0
            for ci in range(n_conns):
                t = topology["Nto1"][ci]
                dim = t["Dimensions"]
                to_rev = int(t["nFurcation"])
                if to_rev == 1:
                    visit_order = [dim] + list(range(1, dim))
                else:
                    visit_order = list(range(1, dim + 1))
                for local_p in visit_order:
                    global_p = prev_ports + local_p
                    if remaining[global_p - 1] != 0:
                        open_ports_list.append(global_p)
                prev_ports += dim

            new_topology: dict[str, Any] = {
                "PortToCondense": {"a": ptc_a, "b": ptc_b},
                "OpenPorts": open_ports_list,
            }
            sout = CondenseGSM(new_gsm, sele_dim, new_topology)

            # Build sinfo for the final open ports
            cum_dims = [0]
            for ci in range(n_conns):
                cum_dims.append(cum_dims[-1] + topology["Nto1"][ci]["Dimensions"])

            final_sinfo: list[dict[str, Any]] = []
            for global_p in open_ports_list:
                # Determine which Nto1 block this port belongs to
                ci = 0
                while ci < n_conns - 1 and global_p > cum_dims[ci + 1]:
                    ci += 1
                local_p = global_p - cum_dims[ci]
                n_s1 = len(nto1_connections[ci]["SideOne"])
                n_s2 = len(nto1_connections[ci]["SideTwo"])
                n_frc = int(topology["Nto1"][ci]["nFurcation"])
                sinfo_idx = _stmp_port_to_sinfo_idx(local_p, n_s1, n_s2, n_frc)
                final_sinfo.append(conn_sinfos[ci][sinfo_idx])
            sinfo = final_sinfo

        sout = RenormalizeGSM(sout, sele_dim, global_open_side_one, global_open_side_two)
        sf.append(sout)

    # Apply DeviceSymmetry mirroring after all frequencies are solved
    if sym_use == 1:
        # Mirror per-frequency: apply on each sf entry
        sf_sym: list[np.ndarray] = []
        sinfo_original = list(sinfo)
        sinfo_new = None  # Will be set once, then reused for all frequencies
        for sout in sf:
            n_half_ports = sout.shape[0] // sele_dim[0]
            new_gsm = _blkdiag(sout, sout)
            if sym_side == 2:
                # Connect port n_half_ports of copy1 with port 2*n_half_ports of copy2
                pa = n_half_ports
                pb = 2 * n_half_ports
                open_p = [p for p in range(1, 2 * n_half_ports + 1) if p not in (pa, pb)]
                sym_top: dict[str, Any] = {"PortToCondense": {"a": [pa], "b": [pb]}, "OpenPorts": open_p}
                # After condensing the junction ports (SideTwo), final ports are the arms and their copies
                # Set up sinfo following the MATLAB code pattern: ports 1,2 from SideOne, ports 3,4 mirror the output
                if sinfo_new is None:
                    sinfo_new = []
                    # Ports from open_p [1,2,4,5] map to ports 1,2,3,4 of result
                    for p in open_p:
                        # For now, use the same mapping as before (arm ports repeated)
                        if p <= n_half_ports - 1:
                            sinfo_new.append(sinfo_original[p - 1])
                        else:
                            sinfo_new.append(sinfo_original[p - n_half_ports - 1])
                    # Following MATLAB pattern: last port should match output port structure
                    # Replicate the coupler output mode structure to the "output" side
                    if len(sinfo_original) > n_half_ports - 1:
                        sinfo_new[n_half_ports - 1] = sinfo_original[-1]
                        if len(sinfo_new) > n_half_ports:
                            sinfo_new[n_half_ports] = sinfo_original[-1]
                # Renormalization using original side structures (but skip for now)
                sym_side_one = global_open_side_one
                sym_side_two = global_open_side_two
            else:  # Side == 1
                # Connect port 1 of copy1 with port n_half_ports+1 of copy2
                pa = 1
                pb = n_half_ports + 1
                open_p = [p for p in range(1, 2 * n_half_ports + 1) if p not in (pa, pb)]
                sym_top = {"PortToCondense": {"a": [pa], "b": [pb]}, "OpenPorts": open_p}
                if sinfo_new is None:
                    sinfo_new = []
                    for p in open_p:
                        if p <= n_half_ports:
                            sinfo_new.append(sinfo_original[p - 1])
                        else:
                            sinfo_new.append(sinfo_original[p - n_half_ports - 1])
                    if len(sinfo_original) > 0:
                        sinfo_new[0] = sinfo_original[-1]
                        if len(sinfo_new) > 1:
                            sinfo_new[1] = sinfo_original[-1]
                sym_side_one = global_open_side_one
                sym_side_two = global_open_side_two
            sout_cond = CondenseGSM(new_gsm, sele_dim, sym_top)
            # Renormalize after condensation
            sout_cond = RenormalizeGSM(sout_cond, sele_dim, sym_side_one, sym_side_two)
            sf_sym.append(sout_cond)
        sinfo = sinfo_new
        sf = sf_sym

    return sf, sinfo, {}, two_port_devices, nto1_connections, []


def MultiPortDevice(
    two_port_devices: list[dict[str, Any]],
    nto1_connections: list[dict[str, Any]],
    open_ports: list[dict[str, Any]],
    connected_ports: list[Any],
    frequency_sweep: dict[str, Any],
    flag: int,
    options: dict[str, Any],
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[Any], dict[str, Any], dict[str, Any]]:
    two_port_devices, nto1_connections, open_ports, connected_ports, frequency_sweep, symmetry, topology, err = MultiPortDeviceValidate(
        two_port_devices,
        nto1_connections,
        open_ports,
        connected_ports,
        frequency_sweep,
        flag,
        options,
    )
    if err.get("fatal"):
        return [], [], two_port_devices, nto1_connections, connected_ports, frequency_sweep, err

    sf, sinfo, err, two_port_devices, nto1_connections, connected_ports = MultiPortDeviceSolve(
        two_port_devices,
        nto1_connections,
        open_ports,
        connected_ports,
        frequency_sweep,
        symmetry,
        topology,
        options,
    )
    return sf, sinfo, two_port_devices, nto1_connections, connected_ports, frequency_sweep, err


def ExtractSingleS(
    stot: np.ndarray,
    sinfo: list[dict[str, Any]],
    out_port: int,
    in_port: int,
    out_type: str,
    out_m: int,
    out_n: int,
    in_type: str,
    in_m: int,
    in_n: int,
) -> complex:
    out_info = sinfo[out_port - 1]
    in_info = sinfo[in_port - 1]
    sele_dim = (len(out_info["mh"]) + len(out_info["me"]), len(in_info["nh"]) + len(in_info["ne"]))
    s_block = ExtractPortS(stot, sele_dim, out_port, in_port)

    try:
        if in_type == "h":
            i = next(idx for idx, (m, n) in enumerate(zip(in_info["mh"], in_info["nh"])) if m == in_m and n == in_n)
        else:
            i = len(in_info["mh"]) + next(idx for idx, (m, n) in enumerate(zip(in_info["me"], in_info["ne"])) if m == in_m and n == in_n)

        if out_type == "h":
            j = next(idx for idx, (m, n) in enumerate(zip(out_info["mh"], out_info["nh"])) if m == out_m and n == out_n)
        else:
            j = len(out_info["mh"]) + next(idx for idx, (m, n) in enumerate(zip(out_info["me"], out_info["ne"])) if m == out_m and n == out_n)
    except StopIteration:
        return 0.0 + 0.0j

    return s_block[j, i]


def GSMDraw(
    f: np.ndarray,
    sf: list[np.ndarray],
    sinfo: list[dict[str, Any]],
    mode_struct: list[tuple[int, int, str, int, int, str, int, int, str]],
    flag: int = 1,
    ylabel: str = "|S| (dB)",
    ylim: tuple[float, float] = (-60.0, 0.0),
    show: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    if flag:
        plt.figure()

    for mode in mode_struct:
        out_port, in_port, out_type, out_m, out_n, in_type, in_m, in_n, _ = mode
        trace = np.asarray(
            [ExtractSingleS(s, sinfo, out_port, in_port, out_type, out_m, out_n, in_type, in_m, in_n) for s in sf],
            dtype=complex,
        )
        safe_mag = np.maximum(np.abs(trace), np.finfo(float).tiny)
        label = f"S{out_port}{in_port}_{{{out_type}{out_m}{out_n}_{{{in_type}{in_m}{in_n}}}}}"
        plt.plot(f, 20.0 * np.log10(safe_mag), label=label)

    plt.legend()
    plt.xlim(float(np.min(f)), float(np.max(f)))
    plt.ylim(*ylim)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(ylabel)
    if show:
        plt.show()


def RelativePhaseDraw(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("RelativePhaseDraw is not implemented yet")


def MultiPortDeviceDraw(
    two_port_devices: list[dict[str, Any]],
    nto1_connections: list[dict[str, Any]],
    flag: int,
    options: dict[str, Any],
) -> None:
    """Draw every TwoPortDevice and N-to-1 connection belonging to a MultiPortDevice definition.
    
    [IN]
        two_port_devices    - cell array of TwoPortDevice structs
        nto1_connections    - cell array of Nto1Connection structs
        flag                - 0 compact draw, 1 exploded draw
        options             - draw options struct 
    
    [OUT]
        (none)              - opens and populates a figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    position = 0.0
    end_position = 0.0
    
    for index in range(len(nto1_connections)):
        current_wgs = []
        nto1_conn = nto1_connections[index]
        
        # Draw SideOne devices
        for i, side_one_dev in enumerate(nto1_conn["SideOne"]):
            d1_idx = side_one_dev["TwoPortDeviceIndex"] - 1  # Convert to 0-indexed
            
            # Check for symmetry mirror drawing
            sym_use = options.get("DeviceSymmetry", {}).get("Use", 0)
            sym_side = options.get("DeviceSymmetry", {}).get("Side", 2)
            if index == 0 and sym_use == 1 and sym_side == 1:
                sym_conn, _, _ = TwoPortDeviceGetPortSegment(two_port_devices[d1_idx]["D"], 1)
                WaveGuideConnectionCapDraw(ax, sym_conn, "m", 0.0)
            
            # Check if device was already solved in previous connection
            current_wgs_copy = two_port_devices[d1_idx]["D"]
            if index > 0:
                for prev_idx in range(index):
                    for alpha, alpha_dev in enumerate(nto1_connections[prev_idx]["SideTwo"]):
                        for beta, beta_dev in enumerate(nto1_conn["SideOne"]):
                            if (alpha_dev["TwoPortDeviceIndex"] == 
                                beta_dev["TwoPortDeviceIndex"]):
                                sym_seg, _, _ = TwoPortDeviceGetPortSegment(two_port_devices[d1_idx]["D"], 2)
                                current_wgs_copy = [sym_seg]
                                current_wgs_copy[0] = dict(current_wgs_copy[0])
                                current_wgs_copy[0]["l"] = 0
            
            # Set zo (starting z position)
            current_wgs_copy = [dict(seg) for seg in current_wgs_copy]
            current_wgs_copy[0]["zo"] = position
            end_position = TwoPortDeviceDraw(ax, current_wgs_copy, flag)
        
        # Draw Nto1 junction
        nto1_conn_copy = dict(nto1_conn)
        nto1_conn_copy["zo"] = end_position
        Nto1DeviceDraw(ax, two_port_devices, nto1_conn_copy)
        position = end_position
        
        # Draw SideTwo devices
        for i, side_two_dev in enumerate(nto1_conn["SideTwo"]):
            d2_idx = side_two_dev["TwoPortDeviceIndex"] - 1  # Convert to 0-indexed
            
            # Set zo for the device
            dev_copy = {"D": [dict(seg) for seg in two_port_devices[d2_idx]["D"]]}
            dev_copy["D"][0]["zo"] = position
            end_position = TwoPortDeviceDraw(ax, dev_copy["D"], flag)
            
            # Draw connection cap between junctions
            if index < len(nto1_connections) - 1:
                last_seg = dev_copy["D"][-1]
                WaveGuideConnectionCapDraw(ax, last_seg, "c", end_position)
                position = end_position
            
            # Check for symmetry mirror drawing
            if (index == len(nto1_connections) - 1 and 
                sym_use == 1 and sym_side == 2):
                sym_conn, _, _ = TwoPortDeviceGetPortSegment(two_port_devices[d2_idx]["D"], 2)
                WaveGuideConnectionCapDraw(ax, sym_conn, "m", end_position)
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1, 1, 2])  # Adjust aspect ratio for visualization
    plt.show()


def TwoPortDeviceDraw(ax: Any, two_port_device: list[dict[str, Any]], explode: int) -> float:
    """Draw one TwoPortDevice either compactly or exploded along z.
    
    [IN]
        ax                  - matplotlib 3D axes
        two_port_device     - cell array describing the device waveguide structure
        explode             - 0 no draw, 1 compact draw, 2 exploded draw
    
    [OUT]
        position            - z-coordinate of the exit port
    """
    ze = 0.0
    z0 = two_port_device[0].get("zo", 0.0)
    
    if explode:
        for seg in two_port_device:
            ze += seg["b"]
        ze = ze / len(two_port_device)
        if z0 < 0:
            z0 = -(ze * len(two_port_device)) + z0
            if len(two_port_device) == 1:
                z0 = z0 - ze / 2
    
    for i in range(len(two_port_device) - 1):
        z0 = z0 + ze / 2
        z0 = ShowSegment(ax, two_port_device[i], z0) + ze / 2
        WaveGuideCapDraw(ax, two_port_device[i], two_port_device[i + 1], z0)
    
    z0 = z0 + ze / 2
    z0 = ShowSegment(ax, two_port_device[-1], z0)
    
    return z0 + ze / 2


def ShowSegment(ax: Any, waveguide_segment: dict[str, Any], z0: float) -> float:
    """Draw one rectangular waveguide segment as a set of 3-D patches.
    
    [IN]
        ax                  - matplotlib 3D axes
        waveguide_segment   - struct containing a, b, l, xo, yo
        z0                  - starting z-position [m]
    
    [OUT]
        z0                  - ending z-position after drawing the segment
    """
    a = waveguide_segment["a"]
    b = waveguide_segment["b"]
    l = waveguide_segment["l"]
    xo = waveguide_segment["xo"]
    yo = waveguide_segment["yo"]
    
    # Bottom face (y-)
    x_bottom = [xo - a / 2, xo + a / 2, xo + a / 2, xo - a / 2]
    y_bottom = [yo - b / 2, yo - b / 2, yo - b / 2, yo - b / 2]
    z_bottom = [z0, z0, z0 + l, z0 + l]
    verts_bottom = [list(zip(x_bottom, y_bottom, z_bottom))]
    ax.plot_surface(*np.meshgrid([xo - a / 2, xo + a / 2], [z0, z0 + l]),
                   np.ones_like(np.meshgrid([xo - a / 2, xo + a / 2], [z0, z0 + l])[0]) * (yo - b / 2),
                   alpha=0.3, color="blue")
    
    # Top face (y+)
    ax.plot_surface(*np.meshgrid([xo - a / 2, xo + a / 2], [z0, z0 + l]),
                   np.ones_like(np.meshgrid([xo - a / 2, xo + a / 2], [z0, z0 + l])[0]) * (yo + b / 2),
                   alpha=0.3, color="blue")
    
    # Left face (x-)
    ax.plot_surface(np.ones_like(np.meshgrid([yo - b / 2, yo + b / 2], [z0, z0 + l])[0]) * (xo - a / 2),
                   *np.meshgrid([yo - b / 2, yo + b / 2], [z0, z0 + l]),
                   alpha=0.3, color="blue")
    
    # Right face (x+)
    ax.plot_surface(np.ones_like(np.meshgrid([yo - b / 2, yo + b / 2], [z0, z0 + l])[0]) * (xo + a / 2),
                   *np.meshgrid([yo - b / 2, yo + b / 2], [z0, z0 + l]),
                   alpha=0.3, color="blue")
    
    return z0 + l


def WaveGuideCapDraw(ax: Any, seg_a: dict[str, Any], seg_b: dict[str, Any], z0: float) -> None:
    """Draw the transverse cap joining two waveguide cross-sections in 3-D.
    
    [IN]
        ax                  - matplotlib 3D axes
        seg_a               - first waveguide segment struct
        seg_b               - second waveguide segment struct
        z0                  - z-position of the cap plane [m]
    """
    a1 = seg_a["a"]
    b1 = seg_a["b"]
    xo1 = seg_a["xo"]
    yo1 = seg_a["yo"]
    
    a2 = seg_b["a"]
    b2 = seg_b["b"]
    xo2 = seg_b["xo"]
    yo2 = seg_b["yo"]
    
    # Define connection points
    x = [xo1 - a1 / 2, xo2 - a2 / 2, xo2 + a2 / 2, xo1 + a1 / 2]
    y = [yo1 - b1 / 2, yo2 - b2 / 2, yo2 + b2 / 2, yo1 + b1 / 2]
    
    # Connection matrix for patch faces
    conn = [
        [[0, 1, 1, 0], [1, 2, 2, 1]],  # Top-left
        [[1, 2, 2, 1], [2, 3, 3, 2]],  # Top-right
        [[2, 3, 3, 2], [3, 0, 0, 3]],  # Bottom-right
        [[3, 0, 0, 3], [0, 1, 1, 0]],  # Bottom-left
    ]
    
    eps = 1e-10
    for face_x, face_y in conn:
        xp = [x[i] for i in face_x]
        yp = [y[i] for i in face_y]
        
        # Check if patch is inside either cross-section
        in1 = (min(xp) >= xo1 - a1 / 2 - eps and max(xp) <= xo1 + a1 / 2 + eps and
               min(yp) >= yo1 - b1 / 2 - eps and max(yp) <= yo1 + b1 / 2 + eps)
        in2 = (min(xp) >= xo2 - a2 / 2 - eps and max(xp) <= xo2 + a2 / 2 + eps and
               min(yp) >= yo2 - b2 / 2 - eps and max(yp) <= yo2 + b2 / 2 + eps)
        
        if in1 or in2:
            zp = [z0, z0, z0, z0]
            ax.plot_surface(np.array([xp[:2], xp[2:]]), np.array([yp[:2], yp[2:]]),
                          np.array([zp[:2], zp[2:]]), alpha=0.5, color="red")


def WaveGuideConnectionCapDraw(ax: Any, two_port_device: dict[str, Any], color: str, z0: float) -> None:
    """Draw the cap patch corresponding to a TwoPortDevice connection face.
    
    [IN]
        ax                  - matplotlib 3D axes
        two_port_device     - segment or device geometry to draw
        color               - patch color specification
        z0                  - z-position of the cap plane [m]
    """
    a1 = two_port_device["a"]
    b1 = two_port_device["b"]
    xo1 = two_port_device["xo"]
    yo1 = two_port_device["yo"]
    
    x = [xo1 - a1 / 2, xo1 + a1 / 2, xo1 + a1 / 2, xo1 - a1 / 2]
    y = [yo1 - b1 / 2, yo1 - b1 / 2, yo1 + b1 / 2, yo1 + b1 / 2]
    z = [z0, z0, z0, z0]
    
    ax.plot_surface(np.array([x[:2], x[2:]]), np.array([y[:2], y[2:]]),
                   np.array([z[:2], z[2:]]), alpha=0.7, color=color, edgecolor="black")


def Nto1DeviceDraw(ax: Any, two_port_devices: list[dict[str, Any]], nto1_connection: dict[str, Any]) -> None:
    """Draw one N-to-1 connection and the attached TwoPortDevices.
    
    [IN]
        ax                  - matplotlib 3D axes
        two_port_devices    - cell array of all TwoPortDevice structs
        nto1_connection     - Nto1Connection struct to draw
    """
    zo = nto1_connection.get("zo", 0.0)
    
    nbr_ports_side_one = len(nto1_connection["SideOne"])
    nbr_ports_side_two = len(nto1_connection["SideTwo"])
    
    # Determine which side is the "many" side for drawing
    if nbr_ports_side_one > nbr_ports_side_two:
        many_side = nto1_connection["SideOne"]
        few_side = nto1_connection["SideTwo"]
        many_count = nbr_ports_side_one
        few_count = nbr_ports_side_two
    else:
        many_side = nto1_connection["SideTwo"]
        few_side = nto1_connection["SideOne"]
        many_count = nbr_ports_side_two
        few_count = nbr_ports_side_one
    
    # Draw the junction as a series of transition patches
    # This is a simplified representation showing the port openings
    colors_many = ["cyan" if nbr_ports_side_one > nbr_ports_side_two else "green" 
                   for _ in range(many_count)]
    colors_few = ["green" if nbr_ports_side_one > nbr_ports_side_two else "cyan" 
                  for _ in range(few_count)]
    
    # Draw all ports
    for i, dev_info in enumerate(many_side):
        d_idx = dev_info["TwoPortDeviceIndex"] - 1
        p_idx = dev_info["TwoPortDevicePort"]
        seg, _, _ = TwoPortDeviceGetPortSegment(two_port_devices[d_idx]["D"], p_idx)
        color = colors_many[i]
        WaveGuideConnectionCapDraw(ax, seg, color, zo)

    for i, dev_info in enumerate(few_side):
        d_idx = dev_info["TwoPortDeviceIndex"] - 1
        p_idx = dev_info["TwoPortDevicePort"]
        seg, _, _ = TwoPortDeviceGetPortSegment(two_port_devices[d_idx]["D"], p_idx)
        color = colors_few[i]
        WaveGuideConnectionCapDraw(ax, seg, color, zo)



def WaveGuideSegmentGetBounding(*args: Any, **kwargs: Any) -> None:
    seg = args[0]
    a, b, xo, yo = WaveGuideSegmentGetCrossSection(seg)
    return xo - a / 2.0, xo + a / 2.0, yo - b / 2.0, yo + b / 2.0


def WaveGuideSegmentGetCrossSection(*args: Any, **kwargs: Any) -> None:
    seg = args[0]
    return seg["a"], seg["b"], seg["xo"], seg["yo"]


def TwoPortDeviceInsertPortSegment(*args: Any, **kwargs: Any) -> None:
    wgs_to_change = args[0]
    new_seg = args[1]
    pos = args[2]

    out = {"D": list(wgs_to_change["D"])}
    if pos == 1:
        out["D"] = [new_seg] + out["D"]
    elif pos == 2:
        out["D"] = out["D"] + [new_seg]
    else:
        raise ValueError("Position must be 1 (prepend) or 2 (append)")
    return out


def _validate_and_insert_intersections(
    segments: list[dict[str, Any]],
    iw_start: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any], int]:
    iw = iw_start
    sym = {"x": 1, "y": 1, "H": 1, "E": 1}
    err: dict[str, Any] = {}

    required = ["a", "b", "xo", "yo", "l", "Nmodes"]
    for idx, s in enumerate(segments):
        for k in required:
            if k not in s:
                return segments, sym, {"fatal": f"missing field {k} in section {idx + 1}"}, iw
        if s["a"] < np.finfo(float).eps:
            return segments, sym, {"fatal": f"invalid a in section {idx + 1}"}, iw
        if s["b"] < np.finfo(float).eps:
            return segments, sym, {"fatal": f"invalid b in section {idx + 1}"}, iw
        if s["l"] < -np.finfo(float).eps:
            return segments, sym, {"fatal": f"invalid l in section {idx + 1}"}, iw
        if s["Nmodes"] < 1:
            return segments, sym, {"fatal": f"invalid Nmodes in section {idx + 1}"}, iw

    a0 = segments[0]["a"]
    b0 = segments[0]["b"]
    xo0 = segments[0]["xo"]
    yo0 = segments[0]["yo"]

    i = 0
    while i < len(segments) - 1:
        s1 = segments[i]
        s2 = segments[i + 1]

        if abs(xo0 - s2["xo"]) > np.finfo(float).eps:
            sym["x"] = 0
        if abs(yo0 - s2["yo"]) > np.finfo(float).eps:
            sym["y"] = 0
        if abs(a0 - s2["a"]) > np.finfo(float).eps:
            sym["H"] = 0
        if abs(b0 - s2["b"]) > np.finfo(float).eps:
            sym["E"] = 0

        xmin1, xmax1 = s1["xo"] - s1["a"] / 2.0, s1["xo"] + s1["a"] / 2.0
        ymin1, ymax1 = s1["yo"] - s1["b"] / 2.0, s1["yo"] + s1["b"] / 2.0
        xmin2, xmax2 = s2["xo"] - s2["a"] / 2.0, s2["xo"] + s2["a"] / 2.0
        ymin2, ymax2 = s2["yo"] - s2["b"] / 2.0, s2["yo"] + s2["b"] / 2.0

        inside12 = xmin1 >= xmin2 and xmax1 <= xmax2 and ymin1 >= ymin2 and ymax1 <= ymax2
        inside21 = xmin1 <= xmin2 and xmax1 >= xmax2 and ymin1 <= ymin2 and ymax1 >= ymax2
        disjoint = xmin1 >= xmax2 or xmax1 <= xmin2 or ymin1 >= ymax2 or ymax1 <= ymin2

        if disjoint:
            return segments, sym, {"fatal": f"segments {i+1} and {i+2} do not intersect"}, iw
        if not inside12 and not inside21:
            xmina = max(xmin1, xmin2)
            xmaxa = min(xmax1, xmax2)
            ymina = max(ymin1, ymin2)
            ymaxa = min(ymax1, ymax2)
            new_seg = {
                "a": xmaxa - xmina,
                "b": ymaxa - ymina,
                "xo": (xmaxa + xmina) / 2.0,
                "yo": (ymaxa + ymina) / 2.0,
                "l": 0.0,
                "Nmodes": int(min(s1["Nmodes"], s2["Nmodes"])),
            }
            segments = segments[: i + 1] + [new_seg] + segments[i + 1 :]
            iw += 1
            i += 1
            continue

        i += 1

    if sym["x"] == 0:
        sym["H"] = 0
    if sym["y"] == 0:
        sym["E"] = 0

    if sym["x"] == 1 and sym["y"] == 1 and sym["H"] == 1 and sym["E"] == 1 and len(segments) > 1:
        return segments, sym, {"fatal": "waveguide segments are all equal"}, iw

    return segments, sym, err, iw


def TwoPortDeviceValidate(*args: Any, **kwargs: Any) -> tuple[Any, Any, dict[str, Any]]:
    wgs = list(args[0])
    if wgs and "zo" not in wgs[0]:
        wgs[0]["zo"] = 0.0

    wgs, sym, err, _iw = _validate_and_insert_intersections(wgs)
    return wgs, sym, err


def Nto1DeviceValidate(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    two_port_devices = args[0]
    conn = args[1]

    side_one = conn["SideOne"]
    side_two = conn["SideTwo"]
    nbr_side_one = len(side_one)
    nbr_side_two = len(side_two)
    nparts = nbr_side_one + nbr_side_two

    if nbr_side_one > nbr_side_two:
        d1 = [int(x["TwoPortDeviceIndex"]) for x in side_one]
        p1 = [int(x["TwoPortDevicePort"]) for x in side_one]
        d2 = [int(x["TwoPortDeviceIndex"]) for x in side_two]
        p2 = [int(x["TwoPortDevicePort"]) for x in side_two]
        segs = [TwoPortDeviceGetPortSegment(two_port_devices[d1[i] - 1]["D"], p1[i])[0] for i in range(nbr_side_one)]
        segs.append(TwoPortDeviceGetPortSegment(two_port_devices[d2[0] - 1]["D"], p2[0])[0])
    else:
        d1 = [int(x["TwoPortDeviceIndex"]) for x in side_two]
        p1 = [int(x["TwoPortDevicePort"]) for x in side_two]
        d2 = [int(x["TwoPortDeviceIndex"]) for x in side_one]
        p2 = [int(x["TwoPortDevicePort"]) for x in side_one]
        segs = [TwoPortDeviceGetPortSegment(two_port_devices[d1[i] - 1]["D"], p1[i])[0] for i in range(nbr_side_two)]
        segs.append(TwoPortDeviceGetPortSegment(two_port_devices[d2[0] - 1]["D"], p2[0])[0])

    validated, sym, err, _iw = _validate_and_insert_intersections(segs)
    if err.get("fatal"):
        return two_port_devices, conn, sym, err

    if len(validated) > nparts:
        interposed = validated[-2]
        two_port_devices[d1[0] - 1] = TwoPortDeviceInsertPortSegment(two_port_devices[d1[0] - 1], interposed, p1[0])

    return two_port_devices, conn, sym, {}


def NotInRect(*args: Any, **kwargs: Any) -> bool:
    xmean, ymean, a, b, xo, yo = args
    x_index = np.argsort(np.asarray(xo))
    found = False
    for i in range(len(x_index) - 1):
        idx1 = x_index[i]
        idx2 = x_index[i + 1]
        xmax = xo[idx1] + a[idx1] / 2.0
        ymax = yo[idx1] + b[idx1] / 2.0
        xmin = xo[idx2] - a[idx2] / 2.0
        ymin = yo[idx2] - b[idx2] / 2.0
        xm = (xmax + xmin) / 2.0
        ym = (ymax + ymin) / 2.0
        if xm == xmean and ym == ymean:
            found = True
            break
    return bool(found)


def DumpError(prefix: str, error: dict[str, Any]) -> bool:
    del prefix
    return bool(error.get("fatal"))
