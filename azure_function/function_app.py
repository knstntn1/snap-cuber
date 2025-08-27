# function_app.py
# Azure Functions (Python v2)
# Endpoints:
#   POST /api/solve
#   POST /api/scramble
#
# States/Traces immer im Input-Format:
#   Keys: F, L, R, B, U, D  (Row-Major 0..8), Werte = Farbnamen

import json
from typing import Dict, List, Tuple, Optional
import azure.functions as func

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

REQUIRED_INPUT_FACES = ["F","L","R","B","U","D"]
KOCIEMBA_FACE_ORDER  = ["U","R","F","D","L","B"]
FACE_ORDER = ["U","R","F","D","L","B"]

def _cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }

def _json_response(body: Dict, status_code: int = 200):
    return func.HttpResponse(
        body=json.dumps(body, ensure_ascii=False),
        status_code=status_code,
        mimetype="application/json",
        headers=_cors_headers(),
    )

def _error(msg: str, status: int = 400):
    return _json_response({"error": msg}, status)

def _validate_schema(state: Dict[str, List[str]]) -> None:
    missing = [f for f in REQUIRED_INPUT_FACES if f not in state]
    if missing:
        raise ValueError(f"Missing faces in JSON: {missing}. Expected are {REQUIRED_INPUT_FACES}.")
    for face, stickers in state.items():
        if not isinstance(stickers, list) or len(stickers) != 9:
            raise ValueError(f"Face '{face}' must be a list with exactly 9 entries (row-major 0..8).")
        if not all(isinstance(c, str) and c for c in stickers):
            raise ValueError(f"Face '{face}' contains invalid color values (all must be non-empty strings).")

def _build_color_to_face_map(state_upper: Dict[str, List[str]]) -> Dict[str, str]:
    color_to_face: Dict[str, str] = {}
    for face in ["U","R","F","D","L","B"]:
        center_color = state_upper[face][4].strip().lower()
        if center_color in color_to_face:
            prev = color_to_face[center_color]
            raise ValueError(f"Center color '{center_color}' appears multiple times ({prev} and {face}).")
        color_to_face[center_color] = face
    if len(color_to_face) != 6:
        raise ValueError("Did not detect exactly 6 different center colors.")
    return color_to_face

def _reorder_to_kociemba(state_upper: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []
    for face in KOCIEMBA_FACE_ORDER:
        out.extend(state_upper[face])
    return out

def _build_facelet_string(state_raw: Dict[str, List[str]]) -> str:
    state_upper: Dict[str, List[str]] = {k.upper(): v for k, v in state_raw.items()}
    _validate_schema(state_upper)
    color_to_face = _build_color_to_face_map(state_upper)
    ordered_colors = _reorder_to_kociemba(state_upper)
    letters = []
    for col in ordered_colors:
        c = col.strip().lower()
        if c not in color_to_face:
            known = ", ".join(sorted(color_to_face.keys()))
            raise ValueError(f"Color '{col}' unknown. Known center colors: {known}")
        letters.append(color_to_face[c])
    facelet_str = "".join(letters)
    if len(facelet_str) != 54:
        raise ValueError(f"Facelet string has length {len(facelet_str)} instead of 54.")
    return facelet_str

# ---------- Cube model ----------
def _rot_cw(a: List[str]) -> List[str]:
    return [a[i] for i in [6,3,0,7,4,1,8,5,2]]
def _rot_ccw(a: List[str]) -> List[str]:
    return [a[i] for i in [2,5,8,1,4,7,0,3,6]]

class Cube:
    def __init__(self, faces: Dict[str, List[str]]):
        self.f = {k: faces[k][:] for k in ["U","R","F","D","L","B"]}

    def copy(self): return Cube({k: self.f[k][:] for k in ["U","R","F","D","L","B"]})

    # face turns (CW)
    def U(self):
        f=self.f; f["U"]=_rot_cw(f["U"])
        f["F"][0],f["F"][1],f["F"][2], f["R"][0],f["R"][1],f["R"][2], f["B"][0],f["B"][1],f["B"][2], f["L"][0],f["L"][1],f["L"][2] = \
        f["L"][0],f["L"][1],f["L"][2], f["F"][0],f["F"][1],f["F"][2], f["R"][0],f["R"][1],f["R"][2], f["B"][0],f["B"][1],f["B"][2]
    def D(self):
        f = self.f
        f["D"] = _rot_cw(f["D"]) 
        F6,F7,F8 = f["F"][6], f["F"][7], f["F"][8]
        R6,R7,R8 = f["R"][6], f["R"][7], f["R"][8]
        B6,B7,B8 = f["B"][6], f["B"][7], f["B"][8]
        L6,L7,L8 = f["L"][6], f["L"][7], f["L"][8]
        f["R"][6], f["R"][7], f["R"][8] = F6, F7, F8
        f["B"][6], f["B"][7], f["B"][8] = R6, R7, R8
        f["L"][6], f["L"][7], f["L"][8] = B6, B7, B8
        f["F"][6], f["F"][7], f["F"][8] = L6, L7, L8
    def R(self):
        f=self.f; f["R"]=_rot_cw(f["R"])
        u2,u5,u8=f["U"][2],f["U"][5],f["U"][8]
        f["U"][2],f["U"][5],f["U"][8]=f["F"][2],f["F"][5],f["F"][8]
        f["F"][2],f["F"][5],f["F"][8]=f["D"][2],f["D"][5],f["D"][8]
        f["D"][2],f["D"][5],f["D"][8]=f["B"][6],f["B"][3],f["B"][0]
        f["B"][6],f["B"][3],f["B"][0]=u2,u5,u8
    def L(self):
        f=self.f; f["L"]=_rot_cw(f["L"])
        u0,u3,u6=f["U"][0],f["U"][3],f["U"][6]
        f["U"][0],f["U"][3],f["U"][6]=f["B"][8],f["B"][5],f["B"][2]
        f["B"][8],f["B"][5],f["B"][2]=f["D"][0],f["D"][3],f["D"][6]
        f["D"][0],f["D"][3],f["D"][6]=f["F"][0],f["F"][3],f["F"][6]
        f["F"][0],f["F"][3],f["F"][6]=u0,u3,u6
    def F(self):
        f=self.f; f["F"]=_rot_cw(f["F"])
        u6,u7,u8=f["U"][6],f["U"][7],f["U"][8]
        f["U"][6],f["U"][7],f["U"][8]=f["L"][8],f["L"][5],f["L"][2]
        f["L"][8],f["L"][5],f["L"][2]=f["D"][2],f["D"][1],f["D"][0]
        f["D"][2],f["D"][1],f["D"][0]=f["R"][0],f["R"][3],f["R"][6]
        f["R"][0],f["R"][3],f["R"][6]=u6,u7,u8
    def B(self):
        f=self.f; f["B"]=_rot_cw(f["B"])
        u0,u1,u2=f["U"][0],f["U"][1],f["U"][2]
        f["U"][0],f["U"][1],f["U"][2]=f["R"][8],f["R"][5],f["R"][2]
        f["R"][8],f["R"][5],f["R"][2]=f["D"][8],f["D"][7],f["D"][6]
        f["D"][8],f["D"][7],f["D"][6]=f["L"][0],f["L"][3],f["L"][6]
        f["L"][0],f["L"][3],f["L"][6]=u0,u1,u2

    def move(self, token: str):
        face = token[0]; suf = token[1:] if len(token)>1 else ""
        times = 1 if suf=="" else (2 if suf=="2" else 3)
        for _ in range(times):
            getattr(self, face)()

    def apply(self, tokens: List[str]):
        for t in tokens: self.move(t)

    def rot_y_cw(self):
        f = self.f
        new = {k: f[k][:] for k in ["U","R","F","D","L","B"]}
        new["F"] = f["L"][:]
        new["L"] = f["B"][:]
        new["B"] = f["R"][:]
        new["R"] = f["F"][:]
        new["U"] = _rot_ccw(f["U"])   
        new["D"] = _rot_cw(f["D"])    
        self.f = new

    def rot_y_ccw(self):
        f = self.f
        new = {k: f[k][:] for k in ["U","R","F","D","L","B"]}
        # Top-View CCW um Y: F <- R <- B <- L <- F
        new["F"] = f["R"][:]
        new["R"] = f["B"][:]
        new["B"] = f["L"][:]
        new["L"] = f["F"][:]
        new["U"] = _rot_cw(f["U"])   
        new["D"] = _rot_ccw(f["D"])  
        self.f = new

    def rot_z_ccw(self):
        f=self.f
        new = {k: f[k][:] for k in ["U","R","F","D","L","B"]}
        new["F"] = _rot_ccw(f["F"])
        new["B"] = _rot_cw(f["B"])
        new["U"] = _rot_ccw(f["R"])
        new["R"] = _rot_ccw(f["D"])
        new["D"] = _rot_ccw(f["L"])
        new["L"] = _rot_ccw(f["U"])
        self.f = new

    def rot_z_cw(self):
        f=self.f
        new = {k: f[k][:] for k in ["U","R","F","D","L","B"]}
        new["F"] = _rot_cw(f["F"])
        new["B"] = _rot_ccw(f["B"])
        new["U"] = _rot_cw(f["L"])
        new["L"] = _rot_cw(f["D"])
        new["D"] = _rot_cw(f["R"])
        new["R"] = _rot_cw(f["U"])
        self.f = new

def _parse_moves(solution: str) -> List[str]:
    return [t for t in solution.strip().split() if t]

def _expand_double_moves(tokens: List[str]) -> List[str]:
    """Expands *2-moves into two single quarter turns. ' and single moves remain single."""
    out: List[str] = []
    for t in tokens:
        if len(t) > 1 and t[-1] == "2":
            out.append(t[0])
            out.append(t[0])
        else:
            out.append(t)
    return out

def _state_as_input_format(cube: Cube, face_to_color: Dict[str, str]) -> Dict[str, List[str]]:
    out = {}
    for face in ["F","L","R","B","U","D"]:
        out[face] = [face_to_color[ch] for ch in cube.f[face]]
    return out

def _simulate_command_states_with_orient(start_cube: Cube, commands: List[str]) -> Tuple[List[Cube], List[str]]:
    locked = False
    c = start_cube.copy()
    out_states: List[Cube] = []
    orient_ops: List[str] = []

    for cmd in commands:
        if cmd == "LOCK":
            locked = True
        elif cmd == "RELEASE":
            locked = False
        elif cmd == "TILT":
            c.rot_z_ccw()
        elif cmd == "ROTATE_CW":
            if locked:
                c.apply(["D'"])     
            else:
                c.rot_y_ccw()
        elif cmd == "ROTATE_CCW":
            if locked:
                c.apply(["D"])      
            else:
                c.rot_y_cw()
        else:
            raise ValueError(f"Unbekanntes Command: {cmd}")

        out_states.append(c.copy())
        orient_ops.append("")
    return out_states, orient_ops


def _apply_inverse_orientation(c: Cube, orient_history: List[str]) -> Cube:
    """Applies the inverse net orientation (from history) to a copy to restore the original perspective."""
    tmp = c.copy()
    for op in reversed([op for op in orient_history if op]):
        if op == "Y":
            tmp.rot_y_ccw()
        elif op == "Y'":
            tmp.rot_y_cw()
        elif op == "Z'":
            tmp.rot_z_cw()
    return tmp

def _plan_to_bring_face_to_D(pos: Dict[str, str], target_face: str) -> Optional[List[str]]:
    from collections import deque
    def apply(op: str, p: Dict[str,str]) -> Dict[str,str]:
        m=dict(p)
        if op=="ROTATE_CCW": cyc={'F':'R','R':'B','B':'L','L':'F'}
        elif op=="ROTATE_CW": cyc={'F':'L','L':'B','B':'R','R':'F'}
        elif op=="TILT":      cyc={'U':'L','L':'D','D':'R','R':'U'}
        q={}
        for k in m: q[k]=cyc.get(m[k], m[k])
        return q
    q=deque([(pos.copy(), [])]); seen={tuple(sorted(pos.items()))}
    for _ in range(24):
        for _ in range(len(q)):
            cur,ops=q.popleft()
            if cur[target_face]=='D': return ops
            for op in ("ROTATE_CCW","ROTATE_CW","TILT"):
                nxt=apply(op,cur)
                key=tuple(sorted(nxt.items()))
                if key in seen: continue
                seen.add(key); q.append((nxt, ops+[op]))
    return None

def _parse_kociemba_pairs(solution: str) -> List[Tuple[str,int]]:
    out=[]
    for t in _parse_moves(solution):
        face=t[0]; suf=t[1:] if len(t)>1 else ""
        q = +1 if suf=="" else (-1 if suf=="'" else (+2 if suf=="2" else None))
        if q is None: raise ValueError(f"Unknown move: {t}")
        out.append((face,q))
    return out

def _map_moves_to_commands(moves: List[Tuple[str,int]]) -> List[str]:
    cmds: List[str] = []
    pos = {f:f for f in "URFDLB"}
    locked = False

    def apply_global(op: str):
        nonlocal locked, pos
        if locked:
            cmds.append("RELEASE"); locked = False
        if op == "ROTATE_CCW":
            cyc = {'F':'R','R':'B','B':'L','L':'F'}
        elif op == "ROTATE_CW":
            cyc = {'F':'L','L':'B','B':'R','R':'F'}
        else:  # TILT
            cyc = {'U':'L','L':'D','D':'R','R':'U'}
        pos = {k: cyc.get(v, v) for k, v in pos.items()}
        cmds.append(op)

    i = 0
    while i < len(moves):
        face, qturns = moves[i]
        t = qturns % 4
        if t == 0:
            i += 1
            continue

        plan = _plan_to_bring_face_to_D(pos.copy(), face)
        if plan is None:
            raise RuntimeError(f"Plan not found for {face}")
        for g in plan:
            apply_global(g)

        if not locked:
            cmds.append("LOCK"); locked = True

        def emit(tt: int):
            if tt == 1:
                cmds.append("ROTATE_CCW")             
            elif tt == 2:
                cmds.extend(["ROTATE_CCW", "ROTATE_CCW"])
            elif tt == 3:
                cmds.append("ROTATE_CW")              

        emit(t)
        j = i + 1
        while j < len(moves) and moves[j][0] == face:
            emit(moves[j][1] % 4)
            j += 1

        cmds.append("RELEASE"); locked = False
        i = j

    return cmds

# ---------- /solve ----------
@app.route(route="solve")
def solve(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse(status_code=204, headers=_cors_headers())

    # read JSON
    try:
        raw = req.get_body()
        if not raw: return _error("Empty request body.", 400)
        payload = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(payload, dict): return _error("JSON body must be an object.", 400)
    except Exception as e:
        return _error(f"Invalid JSON: {e}", 400)

    # build facelet & initial cube (letters) using YOUR color names
    try:
        state_upper = {k.upper(): v for k,v in payload.items()}
        _validate_schema(state_upper)
        color_to_face = _build_color_to_face_map(state_upper)    
        face_to_color = {v:k for k,v in color_to_face.items()}   
        facelet = _build_facelet_string(payload)
        faces_letters = {
            "U": list(facelet[0:9]),
            "R": list(facelet[9:18]),
            "F": list(facelet[18:27]),
            "D": list(facelet[27:36]),
            "L": list(facelet[36:45]),
            "B": list(facelet[45:54]),
        }
        cube0 = Cube(faces_letters)
    except Exception as e:
        return _error(str(e), 400)

    # solve with kociemba
    try:
        import kociemba
    except Exception:
        return _error("Server-side is missing the 'kociemba' package. Please add it to requirements.txt.", 500)

    try:
        solution = kociemba.solve(facelet).strip()
        moves_tokens = _parse_moves(solution)
        # NEU: expandierte Steps (z. B. D2 -> D, D)
        solution_steps = _expand_double_moves(moves_tokens)

        # Kociemba trace (states nach jedem Move)
        kociemba_states = []
        kociemba_trace = []
        c = cube0.copy()
        for t in moves_tokens:
            c.move(t)
            state = _state_as_input_format(c, face_to_color)
            kociemba_states.append(state)
            kociemba_trace.append({"move": t, "state": state})

        # Robot commands + trace (feste Perspektive)
        move_pairs = _parse_kociemba_pairs(solution)
        commands = _map_moves_to_commands(move_pairs)
        cmd_states_cubes, orient_ops = _simulate_command_states_with_orient(cube0, commands)

        command_states = []
        command_trace = []
        history = []
        for cmd, cube_after, tag in zip(commands, cmd_states_cubes, orient_ops):
            if tag:
                history.append(tag)
            norm = _apply_inverse_orientation(cube_after, history)
            state = _state_as_input_format(norm, face_to_color)
            command_states.append(state)
            command_trace.append({"command": cmd, "state": state})

        return _json_response({
            "solution": solution,
            "solution_steps": solution_steps,  
            "length": len(moves_tokens),
            "facelet_string": facelet,
            "commands": commands,
            "commands_count": len(commands),
            "kociemba_states": kociemba_states,   
            "command_states": command_states,     
            "kociemba_trace": kociemba_trace,
            "command_trace": command_trace,
        }, 200)
    except Exception as e:
        return _error(f"Could not solve/map: {e}", 400)

# ---------- /scramble ----------
@app.route(route="scramble")
def scramble(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse(status_code=204, headers=_cors_headers())

    # read JSON: requires starting state + optional len/seed
    try:
        raw = req.get_body()
        if not raw: return _error("Empty request body. Expecting initial state with F/L/R/B/U/D.", 400)
        payload = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(payload, dict): return _error("JSON body must be an object.", 400)
    except Exception as e:
        return _error(f"Invalid JSON: {e}", 400)

    # parameters
    try:
        length = int(payload.get("len", 25))
    except Exception:
        return _error("len must be a number.", 400)
    if not (1 <= length <= 100):
        return _error("len must be between 1 and 100.", 400)
    seed = payload.get("seed")
    seed = int(seed) if seed is not None else None

    # Validate + build cube from provided starting colors
    try:
        state_upper = {k.upper(): v for k,v in payload.items() if k.upper() in REQUIRED_INPUT_FACES}
        _validate_schema(state_upper)
        color_to_face = _build_color_to_face_map(state_upper)    
        face_to_color = {v:k for k,v in color_to_face.items()}   
        facelet = _build_facelet_string(state_upper)
        faces_letters = {
            "U": list(facelet[0:9]),
            "R": list(facelet[9:18]),
            "F": list(facelet[18:27]),
            "D": list(facelet[27:36]),
            "L": list(facelet[36:45]),
            "B": list(facelet[45:54]),
        }
        cube0 = Cube(faces_letters)
        initial_state_colors = _state_as_input_format(cube0, face_to_color)
    except Exception as e:
        return _error(f"Initial state invalid: {e}", 400)

    # Scramble generator
    def gen(len_=25, seed=None):
        import random
        if seed is not None: random.seed(int(seed))
        faces=list("URFDLB"); suf=["","'","2"]
        out=[]; last=None; last_axis=None; axis={'U':'U','D':'U','L':'L','R':'L','F':'F','B':'F'}
        while len(out) < len_:
            f=random.choice(faces); ax=axis[f]
            if f==last or ax==last_axis: continue
            out.append(f+random.choice(suf)); last, last_axis=f, ax
        return out

    # build scramble and traces on top of the provided start cube
    moves = gen(length, seed)

    # Scramble trace (states after each move) using the SAME color names as input
    kociemba_states = []
    kociemba_trace = []
    c = cube0.copy()
    for t in moves:
        c.move(t)
        state = _state_as_input_format(c, face_to_color)
        kociemba_states.append(state)
        kociemba_trace.append({"move": t, "state": state})

    # Commands & command state trace (mapping of scramble moves)
    move_pairs = []
    for t in moves:
        face=t[0]; suf=t[1:] if len(t)>1 else ""
        q = +1 if suf=="" else (-1 if suf=="'" else (+2 if suf=="2" else None))
        move_pairs.append((face,q))
    commands = _map_moves_to_commands(move_pairs)
    cmd_states_cubes, orient_ops = _simulate_command_states_with_orient(cube0, commands)

    command_states = []
    command_trace = []
    history = []
    for cmd, cube_after, tag in zip(commands, cmd_states_cubes, orient_ops):
        if tag:
            history.append(tag)
        norm = _apply_inverse_orientation(cube_after, history)
        state = _state_as_input_format(norm, face_to_color)
        command_states.append(state)
        command_trace.append({"command": cmd, "state": state})

    return _json_response({
        "initial_state": initial_state_colors,  
        "scramble": " ".join(moves),
        "length": len(moves),
        "commands": commands,
        "commands_count": len(commands),
        "kociemba_states": kociemba_states,  
        "command_states": command_states,    
        "kociemba_trace": kociemba_trace,
        "command_trace": command_trace,
    }, 200)