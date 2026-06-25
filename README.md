# UP-DOWN
import { useState, useEffect, useCallback } from "react";

/* ═══════════════════════════════════════ CONSTANTS ══ */
const UP_C  = "#00e5a0", DOWN_C = "#ff4d72", MID_C = "#a78bfa";
const GOLD  = "#f59e0b", BLUE   = "#3b82f6";
const BG    = "#060d1a", CARD   = "#0b1520", BORDER = "#172435";
const STORAGE_KEY = "seq-engine-v5";

const INIT_SEQS   = ["DUUUD","DUUUU","UUDDD","DUDUU","UUUUD","UDUMD","UDDDD","UDDUU","DUUDU"];
const INIT_MOTIFS = ["UUU","DD","DUD","UDD","DUUU"];
const MAX_W=3.0, MIN_W=0.1, REWARD=0.15, PENALTY=0.10;

const INIT_STATE  = {
  // ── standard engine — its own weights ──
  sequences:    [...INIT_SEQS],
  seqWeights:   Object.fromEntries(INIT_SEQS.map(s   => [s,1.00])),
  motifWeights: Object.fromEntries(INIT_MOTIFS.map(m => [m,1.00])),
  frequencies:  { U:25, D:19, M:1 },
  learnedPartials: [],
  history:      [],
  totalPred:    0,
  correctPred:  0,
  // ── M-sequence engine — its own, separate weights ──
  lastAnswer:   null,            // most recent actual outcome recorded, any engine — anchors the flip rule
  mWeight:      1.00,            // Laplace-style reliability weight for "opposite of previous"
  mFrequencies: { U:0, D:0 },
  mPatterns:    {},              // e.g. "U→M→D": 3
  mHistory:     [],
  mTotalPred:   0,
  mCorrectPred: 0,
  // ── cross-engine memory — notes only, never alters classification ──
  crossMatches: [],
};

/* ═══════════════════════════════════════ M-ENGINE LOGIC ══ */
const hasM = syms => syms.includes("M");
const opposite = sym => sym==="U" ? "D" : sym==="D" ? "U" : null;

// Finds the most recent M in the input window and what came right before it.
// If M is the very first symbol, "previous answer" falls back to the engine's
// last globally-recorded actual outcome — this is how M "reaches across"
// sequence boundaries to find what it should flip.
function getMContext(syms, lastAnswer) {
  let mIdx = -1;
  for (let i = syms.length - 1; i >= 0; i--) if (syms[i] === "M") { mIdx = i; break; }
  if (mIdx === -1) return null;
  const prevSym = mIdx > 0 ? syms[mIdx - 1] : lastAnswer;
  return { mIdx, prevSym };
}

function computeFlip(ctx) {
  if (!ctx) return { flip: null, certain: false };
  const o = opposite(ctx.prevSym);
  if (o) return { flip: o, certain: true };
  return { flip: "U", certain: false }; // no usable prior answer — weak default
}

// Cheap, deterministic guess from the OPPOSITE engine — used only for the
// cross-check / memory, never as an official prediction or classification.
function shadowCandidate(kb, primaryType) {
  if (primaryType === "standard") return opposite(kb.lastAnswer);
  const entries = Object.entries(kb.frequencies);
  if (entries.length === 0) return null;
  return entries.sort((a,b)=>b[1]-a[1])[0][0];
}

// Has a similar cross-match happened before, for this engine + context?
function relevantCrossMatches(kb, primaryType, contextKey) {
  return (kb.crossMatches||[]).filter(c => c.primaryType===primaryType && c.contextKey===contextKey);
}

// Fully deterministic — the flip rule must never be left to chance.
function predictMSequence(syms, kb) {
  const ctx = getMContext(syms, kb.lastAnswer);
  const { flip, certain } = computeFlip(ctx);
  const w = kb.mWeight;
  const confidence = Math.round(
    certain ? Math.min(95, Math.max(50, (w/(w+1))*100))
            : Math.min(40, Math.max(20, (w/(w+2))*100))
  );
  const altSym = flip === "U" ? "D" : flip === "D" ? "U" : "D";
  const contextKey = ctx?.prevSym ?? "∅";
  const crossNotes = relevantCrossMatches(kb, "m-sequence", contextKey);
  const reason = certain
    ? `M marks a new sequence boundary. The answer right before M was ${ctx.prevSym}, so the locked opposite-rule prediction is ${flip}. Rule reliability weight is ${w.toFixed(2)} after ${kb.mTotalPred} M-round(s), ${kb.mCorrectPred} correct.${crossNotes.length?` Note: ${crossNotes.length} past round(s) with this same preceding answer also saw the standard engine's bias align with the actual outcome.`:""}`
    : `M detected at the very start of tracked history, so there's no prior answer to flip yet — falling back to a low-confidence default of ${flip}.`;
  return {
    _type: "m-sequence",
    _ctx: ctx,
    _flip: flip,
    _certain: certain,
    _contextKey: contextKey,
    prediction: flip,
    confidence,
    supportingPatterns: [
      ...(certain ? [`M-flip rule: prev=${ctx.prevSym} → opposite=${flip} (weight=${w.toFixed(2)})`] : ["No prior answer available — fallback guess, kept low-confidence"]),
      ...(crossNotes.length ? [`Cross-engine memory: standard-bias matched actual outcome ${crossNotes.length}× before for this same context`] : []),
    ],
    alternatives: [
      { symbol: altSym, confidence: Math.max(5, 100-confidence-10) },
      { symbol: "M",    confidence: Math.max(3, 100-confidence-30) },
    ],
    reason,
  };
}

/* ═══════════════════════════════════════ PROMPT: STANDARD PREDICTION ══ */
function buildPredPrompt(st, contextKey) {
  const tot = st.frequencies.U + st.frequencies.D + st.frequencies.M || 1;
  const pct = s => ((st.frequencies[s]/tot)*100).toFixed(1);
  const acc = st.totalPred>0 ? ((st.correctPred/st.totalPred)*100).toFixed(1) : "0.0";
  const hist = st.history.slice(0,5)
    .map(h=>`  [${(h.input||[]).join("")}]→pred ${h.prediction} actual ${h.actual} ${h.correct?"✓":"✗"}`)
    .join("\n") || "  (none yet)";
  const crossNotes = relevantCrossMatches(st, "standard", contextKey);
  const crossSection = crossNotes.length
    ? `\nCross-engine memory for this exact input [${contextKey}]:\n` +
      crossNotes.slice(0,3).map(c=>`  · actual was ${c.actual}; the M-engine's opposite-of-previous shadow guess (${c.shadowCandidate}) also matched`).join("\n")
    : "";
  return `You are a self-learning sequence prediction engine (STANDARD engine — handles U/D pattern sequences only; M-triggered sequences are handled by a separate flip-rule engine, never by you).
Symbols: U=Up  D=Down  M=Middle

══ LIVE KNOWLEDGE BASE ══
Sequences with learned weights:
${st.sequences.map(s=>`  ${s}  w=${(st.seqWeights[s]||1).toFixed(2)}`).join("\n")}

Motifs with learned weights:
${Object.entries(st.motifWeights).map(([m,w])=>`  ${m}  w=${w.toFixed(2)}`).join("\n")}

Symbol frequencies: U=${st.frequencies.U}(${pct("U")}%) D=${st.frequencies.D}(${pct("D")}%) M=${st.frequencies.M}(${pct("M")}%)
Learning accuracy: ${st.correctPred}/${st.totalPred} (${acc}%)
Recent history:
${hist}
${crossSection}

══ ANALYSIS RULES ══
1. Match the 3-symbol input as a PREFIX against each sequence (first 3 symbols).
   Exact=all 3 match · Near=exactly 2 of 3 match (same index).
2. The 4th symbol from matching sequences is the primary signal.
3. Multiply evidence by current pattern weight. Higher weight = stronger signal.
4. Detect motifs in input: UUU, UU, DD, DU, UD, DUD, UDD, DUUU. Apply motif weights.
5. Break ties with frequency bias. Cross-engine memory above is context only — it must never override your own analysis.
6. If nothing matches, use frequency alone; cap confidence at 40%.

Respond ONLY with valid JSON (no markdown):
{
  "prediction":"U|D|M",
  "confidence":0-100,
  "supportingPatterns":["human-readable strings, e.g. 'DUUUD (exact, w=1.15) → 4th=U'"],
  "alternatives":[{"symbol":"U|D|M","confidence":0-100},{"symbol":"U|D|M","confidence":0-100}],
  "reason":"1-2 sentences citing sequences, weights, motifs, frequencies",
  "matchingSequences":["exact prefix matches"],
  "nearMatches":["e.g. 'DUDUU (pos 1,3)'"],
  "rejectedSequences":["sequences with 0-1 match"],
  "detectedMotifs":["motifs found in input"],
  "rewardKeys":["sequence/motif keys that SUPPORTED this prediction"],
  "penaltyKeys":["keys that ARGUED AGAINST this prediction"]
}`;
}

/* ═══════════════════════════════════════ PROMPT: STANDARD LEARNING ══ */
function buildLearnPrompt(st, input, prediction, actual, rewarded, penalized, newSeqStored, newFreq) {
  return `You are a self-learning sequence engine (STANDARD engine).
A prediction has already been made.

INPUT
Previous Sequence: [${input[0]}][${input[1]}][${input[2]}]
Predicted Symbol: [${prediction}]
Actual Outcome: [${actual}]

KNOWLEDGE BASE CONTEXT
Sequences in database: ${st.sequences.join(", ")}
Motif patterns: ${Object.keys(st.motifWeights).join(", ")}
Old frequencies: U=${st.frequencies.U} D=${st.frequencies.D} M=${st.frequencies.M}
New frequencies: U=${newFreq.U} D=${newFreq.D} M=${newFreq.M}
Patterns rewarded: ${rewarded.length>0 ? rewarded.join(", ") : "none"}
Patterns penalized: ${penalized.length>0 ? penalized.join(", ") : "none"}
New partial sequence [${input.join("")+actual}] stored: ${newSeqStored?"YES":"NO"}

TASK
Write the learning update report in exactly this format.
Be specific — cite the actual pattern names, weights, and numbers.
Only update STANDARD engine weights here — never reference or change M-engine weights.

Respond ONLY with valid JSON (no markdown):
{
  "result":"CORRECT or INCORRECT",
  "patternsRewarded":["list of rewarded patterns with detail, or ['None']"],
  "patternsPenalized":["list of penalized patterns with detail, or ['None']"],
  "updatedFrequencies":{"U":${newFreq.U},"D":${newFreq.D},"M":${newFreq.M}},
  "newSequenceStored":"YES or NO",
  "knowledgeBaseUpdated":"YES",
  "whatChanged":"2-3 sentence explanation of exactly what was learned, which weights moved, why this helps future predictions"
}`;
}

/* ═══════════════════════════════════════ STANDARD LEARNING ENGINE ══ */
function applyWeights(st, prediction, actual, rewardKeys, penaltyKeys, input) {
  const correct = prediction === actual;
  const sw = {...st.seqWeights}, mw = {...st.motifWeights};
  const newFreq = {...st.frequencies, [actual]: (st.frequencies[actual]||0)+1};
  const rewarded=[], penalized=[];

  const adj=(obj,key,delta)=>{
    if(!(key in obj)) return null;
    const before=obj[key];
    obj[key]=Math.min(MAX_W,Math.max(MIN_W,before+delta));
    return `${key} (${before.toFixed(2)}→${obj[key].toFixed(2)})`;
  };

  if(correct) {
    rewardKeys.forEach(k=>{ const r=adj(sw,k,+REWARD)||adj(mw,k,+REWARD); if(r) rewarded.push(r); });
  } else {
    rewardKeys.forEach(k=>{ const r=adj(sw,k,-PENALTY)||adj(mw,k,-PENALTY); if(r) penalized.push(r); });
    penaltyKeys.forEach(k=>{ const r=adj(sw,k,+REWARD*0.5)||adj(mw,k,+REWARD*0.5); if(r) rewarded.push(r); });
  }

  const partial = (input||[]).join("")+actual;
  const alreadyKnown = st.sequences.some(s=>s.startsWith(partial)) ||
                       st.learnedPartials.includes(partial);
  const newLearnedPartials = alreadyKnown ? st.learnedPartials : [...st.learnedPartials, partial];

  const newHistory = [{ input: input||[], prediction, actual, correct, ts: Date.now() }, ...st.history].slice(0,50);

  return {
    newState: {
      ...st, seqWeights: sw, motifWeights: mw, frequencies: newFreq,
      learnedPartials: newLearnedPartials, history: newHistory,
      correctPred: st.correctPred+(correct?1:0), totalPred: st.totalPred+1,
    },
    rewarded, penalized, correct, newFreq, newSeqStored: !alreadyKnown, partial,
  };
}

/* ═══════════════════════════════════════ HELPERS ══ */
const symColor = s=>s==="U"?UP_C:s==="D"?DOWN_C:s==="M"?MID_C:"#475569";
const symRgb   = s=>s==="U"?"0,229,160":s==="D"?"255,77,114":s==="M"?"167,139,250":"71,85,105";
const symFull  = s=>s==="U"?"UP":s==="D"?"DOWN":s==="M"?"MID":"—";

/* ═══════════════════════════════════════ API CALL ══ */
async function callClaude(system, userContent) {
  const res = await fetch("https://api.anthropic.com/v1/messages",{
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({ model:"claude-sonnet-4-6", max_tokens:1000,
      system, messages:[{role:"user",content:userContent}] })
  });
  const data = await res.json();
  const text = (data.content||[]).filter(b=>b.type==="text").map(b=>b.text).join("");
  return JSON.parse(text.replace(/```json|```/g,"").trim());
}

/* ═══════════════════════════════════════ UI COMPONENTS ══ */
function ConfBar({pct,color,h=6}) {
  return (
    <div style={{height:h,background:"#0a1422",borderRadius:h/2,overflow:"hidden"}}>
      <div style={{width:`${Math.min(pct,100)}%`,height:"100%",background:color,
        borderRadius:h/2,transition:"width .7s cubic-bezier(.4,0,.2,1)",
        boxShadow:`0 0 8px ${color}55`}}/>
    </div>
  );
}

function SymKey({sym,active,onClick,disabled}) {
  const c=symColor(sym),r=symRgb(sym);
  return (
    <button onClick={disabled?undefined:onClick} style={{
      flex:1,padding:"6px 2px",borderRadius:6,fontSize:12,
      fontFamily:"'Fira Code',monospace",fontWeight:700,
      cursor:disabled?"not-allowed":"pointer",
      background:active?`rgba(${r},.18)`:"rgba(255,255,255,.025)",
      border:`1px solid ${active?c+"55":BORDER}`,
      color:active?c:disabled?"#1a2a38":"#334155",
      boxShadow:active?`0 0 8px rgba(${r},.28)`:"none",
      transition:"all .13s",opacity:disabled?.5:1,
    }}>{sym}</button>
  );
}

function SeqChip({seq,type,weight}) {
  const c  =type==="exact"?UP_C:type==="near"?GOLD:type==="rejected"?"#1e3048":"#253448";
  const rgb=type==="exact"?"0,229,160":type==="near"?"245,158,11":"37,52,72";
  return (
    <div style={{display:"inline-flex",alignItems:"center",gap:3,flexWrap:"nowrap",
      background:`rgba(${rgb},.1)`,border:`1px solid rgba(${rgb},.28)`,
      borderRadius:6,padding:"3px 8px"}}>
      {seq.split("").map((ch,i)=>(
        <span key={i} style={{color:symColor(ch),fontFamily:"'Fira Code',monospace",
          fontSize:12,fontWeight:700}}>{ch}</span>
      ))}
      {weight!==undefined&&(
        <span style={{color:weight>=1.2?UP_C:weight<=0.8?DOWN_C:GOLD,
          fontFamily:"'Fira Code',monospace",fontSize:9,marginLeft:3}}>
          w={weight.toFixed(2)}
        </span>
      )}
    </div>
  );
}

/* ── KBPanel ── */
function KBPanel({state}) {
  const tot = state.frequencies.U+state.frequencies.D+state.frequencies.M||1;
  const acc = state.totalPred>0 ? ((state.correctPred/state.totalPred)*100).toFixed(0) : "—";
  const accColor = state.totalPred===0 ? "#334155"
    : state.correctPred/state.totalPred>=0.6 ? UP_C
    : state.correctPred/state.totalPred>=0.4 ? GOLD : DOWN_C;

  const SL = ({children, color="#3b82f6"}) => (
    <div style={{display:"flex",alignItems:"center",gap:7,marginBottom:8}}>
      <div style={{width:2,height:12,background:color,borderRadius:1}}/>
      <span style={{color,fontSize:9,fontFamily:"'Fira Code',monospace",
        letterSpacing:"0.09em",textTransform:"uppercase",fontWeight:600}}>{children}</span>
    </div>
  );

  return (
    <div style={{background:CARD,border:`1px solid ${BORDER}`,borderRadius:11,overflow:"hidden"}}>

      {/* ── HEADER ── */}
      <div style={{background:"linear-gradient(135deg,#0e1f38 0%,#0b1520 100%)",
        borderBottom:`1px solid ${BORDER}`,padding:"12px 14px"}}>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:8}}>
          <span style={{fontFamily:"'Syne',sans-serif",fontSize:14,fontWeight:800,
            color:"#e2e8f0",letterSpacing:"-0.01em"}}>Knowledge Base</span>
          <div style={{display:"flex",gap:3}}>
            {[UP_C,MID_C,DOWN_C].map(c=>(
              <div key={c} style={{width:3,height:14,background:c,borderRadius:1.5,opacity:.7}}/>
            ))}
          </div>
        </div>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <div style={{flex:1,height:5,background:"#0a1422",borderRadius:3,overflow:"hidden"}}>
            <div style={{
              width:`${state.totalPred>0?(state.correctPred/state.totalPred*100):0}%`,
              height:"100%",background:accColor,borderRadius:3,transition:"width .6s"}}/>
          </div>
          <span style={{color:accColor,fontFamily:"'Fira Code',monospace",fontSize:11,fontWeight:700,whiteSpace:"nowrap"}}>
            {state.correctPred}/{state.totalPred} ({acc}%)
          </span>
        </div>
      </div>

      <div style={{padding:"14px 14px"}}>

        {/* ── FREQUENCIES ── */}
        <SL color={BLUE}>Symbol Frequencies</SL>
        <div style={{marginBottom:14}}>
          {[
            {s:"U", c:UP_C,   rgb:"0,229,160"},
            {s:"D", c:DOWN_C, rgb:"255,77,114"},
            {s:"M", c:MID_C,  rgb:"167,139,250"},
          ].map(({s,c,rgb})=>{
            const count = state.frequencies[s];
            const pct   = (count/tot*100).toFixed(1);
            return (
              <div key={s} style={{
                background:`rgba(${rgb},.07)`,border:`1px solid rgba(${rgb},.2)`,
                borderRadius:7,padding:"7px 10px",marginBottom:5,
                display:"flex",alignItems:"center",gap:8}}>
                <span style={{color:c,fontFamily:"'Fira Code',monospace",
                  fontSize:16,fontWeight:800,width:16}}>{s}</span>
                <div style={{flex:1,height:5,background:"#0a1422",borderRadius:3,overflow:"hidden"}}>
                  <div style={{width:`${pct}%`,height:"100%",background:c,
                    borderRadius:3,boxShadow:`0 0 6px ${c}60`}}/>
                </div>
                <span style={{color:c,fontFamily:"'Fira Code',monospace",
                  fontSize:12,fontWeight:700,minWidth:20,textAlign:"right"}}>{count}</span>
                <span style={{color:`rgba(${rgb},.5)`,fontFamily:"'Fira Code',monospace",
                  fontSize:10,minWidth:36}}>{pct}%</span>
              </div>
            );
          })}
        </div>

        {/* ── SEQUENCE WEIGHTS ── */}
        <SL color={GOLD}>Standard Sequence Weights</SL>
        <div style={{display:"flex",flexDirection:"column",gap:4,marginBottom:14}}>
          {state.sequences.map(seq=>{
            const w   = state.seqWeights[seq]||1;
            const wc  = w>=1.2?UP_C:w<=0.8?DOWN_C:GOLD;
            const wRgb= w>=1.2?"0,229,160":w<=0.8?"255,77,114":"245,158,11";
            const bar = ((w-0.1)/(3-0.1)*100);
            return (
              <div key={seq} style={{
                background:`rgba(${wRgb},.06)`,border:`1px solid rgba(${wRgb},.18)`,
                borderRadius:7,padding:"6px 9px",display:"flex",alignItems:"center",gap:7}}>
                <span style={{fontFamily:"'Fira Code',monospace",fontSize:12,
                  letterSpacing:"0.05em",minWidth:52}}>
                  {seq.split("").map((ch,i)=>(
                    <span key={i} style={{color:symColor(ch),fontWeight:700}}>{ch}</span>
                  ))}
                </span>
                <div style={{flex:1,height:4,background:"#0a1422",borderRadius:2,overflow:"hidden"}}>
                  <div style={{width:`${bar}%`,height:"100%",background:wc,
                    borderRadius:2,transition:"width .5s",
                    boxShadow:`0 0 4px ${wc}60`}}/>
                </div>
                <span style={{
                  color:wc,fontFamily:"'Fira Code',monospace",fontSize:10,fontWeight:700,
                  background:`rgba(${wRgb},.12)`,borderRadius:4,padding:"1px 6px",
                  minWidth:36,textAlign:"center"}}>
                  {w.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>

        {/* ── MOTIF WEIGHTS ── */}
        <SL color="#64748b">Motif Weights</SL>
        <div style={{display:"flex",flexWrap:"wrap",gap:5,marginBottom:16}}>
          {Object.entries(state.motifWeights).map(([m,w])=>{
            const wc  = w>=1.2?UP_C:w<=0.8?DOWN_C:"#94a3b8";
            const wRgb= w>=1.2?"0,229,160":w<=0.8?"255,77,114":"148,163,184";
            return (
              <div key={m} style={{
                background:`rgba(${wRgb},.1)`,border:`1px solid rgba(${wRgb},.3)`,
                borderRadius:6,padding:"5px 9px",display:"flex",flexDirection:"column",
                alignItems:"center",gap:2}}>
                <span style={{color:wc,fontFamily:"'Fira Code',monospace",
                  fontSize:13,fontWeight:800,letterSpacing:"0.05em"}}>{m}</span>
                <span style={{color:`rgba(${wRgb},.8)`,fontFamily:"'Fira Code',monospace",
                  fontSize:9,fontWeight:600}}>{w.toFixed(2)}</span>
              </div>
            );
          })}
        </div>

        {/* ── M-SEQUENCE ENGINE (separate KB, separate weights) ── */}
        <div style={{
          background:"rgba(167,139,250,.04)",border:`1px solid rgba(167,139,250,.22)`,
          borderRadius:9,padding:"11px 11px 13px",marginBottom:14}}>
          <SL color={MID_C}>M-Sequence Engine · separate weights</SL>

          <div style={{
            background:"rgba(167,139,250,.08)",border:"1px solid rgba(167,139,250,.25)",
            borderRadius:7,padding:"7px 9px",marginBottom:7,
            display:"flex",alignItems:"center",gap:8}}>
            <span style={{color:MID_C,fontFamily:"'Fira Code',monospace",fontSize:10.5,fontWeight:700,minWidth:72}}>
              flip weight
            </span>
            <div style={{flex:1,height:5,background:"#0a1422",borderRadius:3,overflow:"hidden"}}>
              <div style={{width:`${((state.mWeight-0.1)/(3-0.1))*100}%`,height:"100%",background:MID_C,
                borderRadius:3,boxShadow:`0 0 6px ${MID_C}60`}}/>
            </div>
            <span style={{color:MID_C,fontFamily:"'Fira Code',monospace",fontSize:11.5,fontWeight:700,minWidth:36,textAlign:"right"}}>
              {state.mWeight.toFixed(2)}
            </span>
          </div>

          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:9,flexWrap:"wrap"}}>
            <span style={{color:"#64748b",fontFamily:"'Fira Code',monospace",fontSize:10.5}}>
              accuracy: <span style={{color:MID_C,fontWeight:700}}>
                {state.mCorrectPred}/{state.mTotalPred}
                {state.mTotalPred>0 && ` (${((state.mCorrectPred/state.mTotalPred)*100).toFixed(0)}%)`}
              </span>
            </span>
            <span style={{color:"#64748b",fontFamily:"'Fira Code',monospace",fontSize:10.5,marginLeft:"auto"}}>
              <span style={{color:UP_C,fontWeight:700}}>U={state.mFrequencies.U}</span>{"  "}
              <span style={{color:DOWN_C,fontWeight:700}}>D={state.mFrequencies.D}</span>
            </span>
          </div>

          {Object.keys(state.mPatterns).length>0 ? (
            <div style={{display:"flex",flexWrap:"wrap",gap:4,marginBottom: state.mHistory.length>0 ? 10 : 0}}>
              {Object.entries(state.mPatterns).map(([p,c])=>(
                <div key={p} style={{
                  background:"rgba(167,139,250,.1)",border:"1px solid rgba(167,139,250,.28)",
                  borderRadius:5,padding:"3px 8px",
                  color:MID_C,fontFamily:"'Fira Code',monospace",fontSize:10.5,fontWeight:600}}>
                  {p} ×{c}
                </div>
              ))}
            </div>
          ) : (
            <p style={{color:"#1e3048",fontSize:10.5,fontFamily:"'Fira Code',monospace",marginBottom: state.mHistory.length>0?10:0}}>
              No M-sequences recorded yet
            </p>
          )}

          {state.mHistory.length>0 && (
            <div style={{display:"flex",flexDirection:"column",gap:4}}>
              {state.mHistory.slice(0,4).map((h,i)=>{
                const rc=h.correct?UP_C:DOWN_C, rRgb=h.correct?"0,229,160":"255,77,114";
                return (
                  <div key={i} style={{
                    background:`rgba(${rRgb},.05)`,border:`1px solid rgba(${rRgb},.15)`,
                    borderRadius:6,padding:"4px 8px",display:"flex",alignItems:"center",gap:6}}>
                    <span style={{color:"#475569",fontFamily:"'Fira Code',monospace",fontSize:10}}>prev</span>
                    <span style={{color:symColor(h.prevSym||""),fontFamily:"'Fira Code',monospace",fontSize:11,fontWeight:700}}>
                      {h.prevSym||"∅"}
                    </span>
                    <span style={{color:"#1e3048",fontSize:9}}>↦flip↦</span>
                    <span style={{color:symColor(h.flip||""),fontFamily:"'Fira Code',monospace",fontSize:11,fontWeight:700}}>
                      {h.flip||"?"}
                    </span>
                    <span style={{color:"#1e3048",fontSize:9}}>vs</span>
                    <span style={{color:symColor(h.actual),fontFamily:"'Fira Code',monospace",fontSize:11,fontWeight:700}}>
                      {h.actual}
                    </span>
                    <span style={{marginLeft:"auto",color:rc,fontFamily:"'Fira Code',monospace",fontSize:11,fontWeight:800,
                      background:`rgba(${rRgb},.12)`,borderRadius:4,padding:"1px 6px"}}>{h.correct?"✓":"✗"}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* ── CROSS-ENGINE NOTES — memory only, never reclassifies ── */}
        {state.crossMatches && state.crossMatches.length>0 && (
          <div style={{marginBottom:14}}>
            <SL color={GOLD}>Cross-Engine Notes ({state.crossMatches.length})</SL>
            <div style={{display:"flex",flexDirection:"column",gap:4}}>
              {state.crossMatches.slice(0,4).map((c,i)=>(
                <div key={i} style={{
                  background:"rgba(245,158,11,.06)",border:"1px solid rgba(245,158,11,.2)",
                  borderRadius:6,padding:"5px 9px",display:"flex",alignItems:"center",gap:6,flexWrap:"wrap"}}>
                  <span style={{color:GOLD,fontFamily:"'Fira Code',monospace",fontSize:9.5,fontWeight:700,
                    background:"rgba(245,158,11,.14)",borderRadius:4,padding:"1px 5px"}}>
                    {c.primaryType==="standard"?"STD":"M"} round
                  </span>
                  <span style={{color:"#64748b",fontFamily:"'Fira Code',monospace",fontSize:10}}>
                    ctx=[{c.contextKey}] actual=<span style={{color:symColor(c.actual),fontWeight:700}}>{c.actual}</span>
                    {" · "}{c.shadowType==="standard"?"std bias":"M-flip"} agreed
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── LEARNED PARTIALS ── */}
        {state.learnedPartials.length>0&&(
          <div style={{marginBottom:14}}>
            <SL color={BLUE}>Learned Partials ({state.learnedPartials.length})</SL>
            <div style={{display:"flex",flexWrap:"wrap",gap:4}}>
              {state.learnedPartials.map(p=>(
                <div key={p} style={{
                  background:"rgba(59,130,246,.1)",border:"1px solid rgba(59,130,246,.28)",
                  borderRadius:5,padding:"3px 8px",display:"inline-flex",gap:0}}>
                  {p.split("").map((ch,i)=>(
                    <span key={i} style={{color:i<3?symColor(ch):`rgba(${symRgb(ch)},.6)`,
                      fontFamily:"'Fira Code',monospace",fontSize:11,fontWeight:700}}>{ch}</span>
                  ))}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── RECENT HISTORY (standard) ── */}
        {state.history.length>0&&(
          <div>
            <SL color="#64748b">Recent Standard History</SL>
            <div style={{display:"flex",flexDirection:"column",gap:4}}>
              {state.history.slice(0,5).map((h,i)=>{
                const correct=h.correct;
                const rc=correct?UP_C:DOWN_C, rRgb=correct?"0,229,160":"255,77,114";
                return (
                  <div key={i} style={{
                    background:`rgba(${rRgb},.05)`,border:`1px solid rgba(${rRgb},.15)`,
                    borderRadius:6,padding:"5px 9px",
                    display:"flex",alignItems:"center",gap:7}}>
                    <span style={{fontFamily:"'Fira Code',monospace",fontSize:11,minWidth:30}}>
                      {(h.input||[]).map((ch,j)=>(
                        <span key={j} style={{color:symColor(ch),fontWeight:700}}>{ch}</span>
                      ))||<span style={{color:"#1e3048"}}>???</span>}
                    </span>
                    <span style={{color:"#1e3048",fontSize:10}}>→</span>
                    <span style={{color:symColor(h.prediction),fontFamily:"'Fira Code',monospace",
                      fontSize:11,fontWeight:700}}>{h.prediction}</span>
                    <span style={{color:"#1e3048",fontSize:9}}>|</span>
                    <span style={{color:symColor(h.actual),fontFamily:"'Fira Code',monospace",
                      fontSize:11,fontWeight:700}}>{h.actual}</span>
                    <span style={{marginLeft:"auto",color:rc,fontFamily:"'Fira Code',monospace",
                      fontSize:11,fontWeight:800,
                      background:`rgba(${rRgb},.12)`,borderRadius:4,
                      padding:"1px 6px"}}>{correct?"✓":"✗"}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Prediction Output ── */
function PredOutput({result}) {
  if(!result) return null;
  const pred=result.prediction, pc=symColor(pred), pr=symRgb(pred);
  const isM = result._type === "m-sequence";
  const tagColor = isM ? MID_C : BLUE;
  const lbl2={color:"#2a3d55",fontFamily:"'Fira Code',monospace",fontSize:11,
    fontWeight:600,minWidth:120};
  const row2={display:"flex",alignItems:"flex-start",gap:8,
    paddingBottom:11,borderBottom:`1px solid ${BORDER}`,marginBottom:11};
  return (
    <div style={{background:`rgba(${pr},.04)`,border:`1px solid rgba(${pr},.3)`,
      borderRadius:11,overflow:"hidden",marginBottom:14}}>
      <div style={{background:`linear-gradient(90deg,rgba(${pr},.1),transparent 60%)`,
        borderBottom:`1px solid rgba(${pr},.15)`,padding:"8px 16px",
        display:"flex",alignItems:"center",gap:10,flexWrap:"wrap"}}>
        <span style={{color:"#1e3048",fontSize:9.5,fontFamily:"'Fira Code',monospace",
          letterSpacing:"0.08em",textTransform:"uppercase"}}>
          Prediction Output
        </span>
        <span style={{color:tagColor,fontSize:9,fontFamily:"'Fira Code',monospace",
          fontWeight:700,letterSpacing:"0.05em",textTransform:"uppercase",
          background:`rgba(${isM?"167,139,250":"59,130,246"},.12)`,
          borderRadius:4,padding:"1px 6px"}}>
          {isM ? "M-ENGINE · flip rule" : "STANDARD ENGINE"}
        </span>
      </div>
      <div style={{padding:"16px 18px 14px"}}>
        <div style={row2}>
          <span style={lbl2}>Prediction:</span>
          <div style={{display:"flex",alignItems:"baseline",gap:10}}>
            <span style={{fontFamily:"'Syne',sans-serif",fontSize:56,fontWeight:800,
              color:pc,lineHeight:1,textShadow:`0 0 40px ${pc}50`}}>{pred}</span>
            <span style={{color:pc,fontFamily:"'Fira Code',monospace",fontSize:13,opacity:.7}}>
              {symFull(pred)}
            </span>
          </div>
        </div>
        <div style={row2}>
          <span style={lbl2}>Confidence:</span>
          <div style={{flex:1,minWidth:160}}>
            <div style={{display:"flex",alignItems:"baseline",gap:8,marginBottom:5}}>
              <span style={{color:pc,fontFamily:"'Syne',sans-serif",fontSize:24,fontWeight:800}}>
                {result.confidence}%
              </span>
              <span style={{color:"#253448",fontSize:11,fontFamily:"'Fira Code',monospace"}}>
                {result.confidence>=70?"high":result.confidence>=45?"moderate":"low"}
              </span>
            </div>
            <ConfBar pct={result.confidence} color={pc} h={6}/>
          </div>
        </div>
        <div style={row2}>
          <span style={lbl2}>Supporting<br/>Patterns:</span>
          <div style={{display:"flex",flexDirection:"column",gap:3}}>
            {(result.supportingPatterns||[]).length>0
              ? result.supportingPatterns.map((p,i)=>(
                  <span key={i} style={{color:"#64748b",fontSize:11,
                    fontFamily:"'Fira Code',monospace"}}>· {p}</span>
                ))
              : <span style={{color:"#1e3048",fontSize:11,fontFamily:"'Fira Code',monospace"}}>
                  None — using frequency bias only
                </span>
            }
          </div>
        </div>
        <div style={row2}>
          <span style={lbl2}>Alternative<br/>Predictions:</span>
          <div style={{display:"flex",gap:7,flexWrap:"wrap"}}>
            {(result.alternatives||[]).map((alt,i)=>{
              const ac=symColor(alt.symbol),ar=symRgb(alt.symbol);
              return (
                <div key={i} style={{background:"#070d19",border:`1px solid rgba(${ar},.25)`,
                  borderRadius:7,padding:"7px 10px",minWidth:80}}>
                  <p style={{color:"#1e3048",fontSize:8.5,fontFamily:"'Fira Code',monospace",
                    letterSpacing:"0.06em",textTransform:"uppercase",marginBottom:3}}>
                    Option {i+1}
                  </p>
                  <div style={{display:"flex",alignItems:"baseline",gap:5,marginBottom:4}}>
                    <span style={{color:ac,fontFamily:"'Syne',sans-serif",fontSize:20,fontWeight:800}}>
                      {alt.symbol}
                    </span>
                    <span style={{color:"#253448",fontFamily:"'Fira Code',monospace",fontSize:10}}>
                      {alt.confidence}%
                    </span>
                  </div>
                  <ConfBar pct={alt.confidence} color={ac} h={3}/>
                </div>
              );
            })}
          </div>
        </div>
        <div style={{display:"flex",alignItems:"flex-start",gap:8}}>
          <span style={lbl2}>Reason:</span>
          <p style={{color:"#64748b",fontSize:12,lineHeight:1.7,flex:1}}>{result.reason}</p>
        </div>
      </div>
    </div>
  );
}

/* ── Learning Report ── */
function LearnReport({report, partial, engineType}) {
  if(!report) return null;
  const correct = report.result==="CORRECT";
  const rc=correct?UP_C:DOWN_C, rr=correct?"0,229,160":"255,77,114";
  const isM = engineType === "m-sequence";
  const crossMatch = (report.crossCheck||"").startsWith("MATCH");

  const Field=({label,children,border=true})=>(
    <div style={{display:"flex",alignItems:"flex-start",gap:8,
      ...(border?{paddingBottom:11,borderBottom:`1px solid ${BORDER}`,marginBottom:11}:{})}}>
      <span style={{color:"#2a3d55",fontFamily:"'Fira Code',monospace",fontSize:11,
        fontWeight:600,minWidth:160,flexShrink:0}}>{label}</span>
      <div>{children}</div>
    </div>
  );

  return (
    <div style={{background:`rgba(${rr},.04)`,border:`1px solid rgba(${rr},.3)`,
      borderRadius:11,overflow:"hidden",marginBottom:14}}>
      <div style={{background:`linear-gradient(90deg,rgba(${rr},.12),transparent 60%)`,
        borderBottom:`1px solid rgba(${rr},.2)`,padding:"8px 16px",
        display:"flex",alignItems:"center",gap:10,flexWrap:"wrap"}}>
        <span style={{color:"#1e3048",fontSize:9.5,fontFamily:"'Fira Code',monospace",
          letterSpacing:"0.08em",textTransform:"uppercase"}}>
          Learning Update
        </span>
        <span style={{color:isM?MID_C:BLUE,fontSize:9,fontFamily:"'Fira Code',monospace",
          fontWeight:700,letterSpacing:"0.05em",textTransform:"uppercase",
          background:`rgba(${isM?"167,139,250":"59,130,246"},.12)`,
          borderRadius:4,padding:"1px 6px"}}>
          {isM ? "M-ENGINE · flip rule" : "STANDARD ENGINE"}
        </span>
      </div>
      <div style={{padding:"16px 18px 14px"}}>

        <Field label="Result:">
          <span style={{fontFamily:"'Syne',sans-serif",fontSize:24,fontWeight:800,color:rc}}>
            {report.result}
          </span>
        </Field>

        <Field label="Sequence Type:">
          <span style={{fontFamily:"'Fira Code',monospace",fontSize:12.5,fontWeight:700,
            color:isM?MID_C:BLUE}}>
            {report.classification}
          </span>
        </Field>

        <Field label="Patterns Rewarded:">
          <div style={{display:"flex",flexDirection:"column",gap:3}}>
            {(report.patternsRewarded||["None"]).map((r,i)=>(
              <span key={i} style={{color:r==="None"?"#1e3048":UP_C,fontSize:12,
                fontFamily:"'Fira Code',monospace"}}>
                {r==="None"?"None":("+ "+r)}
              </span>
            ))}
          </div>
        </Field>

        <Field label="Patterns Penalized:">
          <div style={{display:"flex",flexDirection:"column",gap:3}}>
            {(report.patternsPenalized||["None"]).map((p,i)=>(
              <span key={i} style={{color:p==="None"?"#1e3048":DOWN_C,fontSize:12,
                fontFamily:"'Fira Code',monospace"}}>
                {p==="None"?"None":("− "+p)}
              </span>
            ))}
          </div>
        </Field>

        <Field label="Opposite-Seq Check:">
          <span style={{fontFamily:"'Fira Code',monospace",fontSize:11.5,lineHeight:1.65,
            color:crossMatch?GOLD:"#475569"}}>
            {report.crossCheck}
          </span>
        </Field>

        <Field label="Updated Frequencies:">
          <div style={{display:"flex",gap:14,flexWrap:"wrap"}}>
            {["U","D","M"].map(s=>(
              <span key={s} style={{fontFamily:"'Fira Code',monospace",fontSize:13}}>
                <span style={{color:symColor(s),fontWeight:700}}>{s}</span>
                <span style={{color:"#334155"}}> = {(report.updatedFrequencies||{})[s]??0}</span>
              </span>
            ))}
          </div>
        </Field>

        <Field label={isM ? "M-Pattern Stored:" : "New Sequence Stored:"}>
          <div>
            <span style={{color:report.newSequenceStored==="YES"?(isM?MID_C:BLUE):"#334155",
              fontFamily:"'Fira Code',monospace",fontSize:14,fontWeight:700}}>
              {report.newSequenceStored}
            </span>
            {report.newSequenceStored==="YES"&&partial&&(
              <span style={{color:isM?MID_C:BLUE,fontFamily:"'Fira Code',monospace",
                fontSize:11,marginLeft:10,opacity:.8}}>
                [{partial}]
              </span>
            )}
          </div>
        </Field>

        <Field label="Knowledge Base Updated:">
          <span style={{color:UP_C,fontFamily:"'Fira Code',monospace",
            fontSize:14,fontWeight:700}}>{report.knowledgeBaseUpdated} ✓</span>
        </Field>

        <Field label="What Changed:" border={false}>
          <p style={{color:"#64748b",fontSize:12,lineHeight:1.75}}>{report.whatChanged}</p>
        </Field>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════ MAIN ══ */
export default function SequenceEngine() {
  const [kb,     setKb]     = useState(INIT_STATE);
  const [loaded, setLoaded] = useState(false);
  const [syms,   setSyms]   = useState([null,null,null]);
  const [actual, setActual] = useState(null);
  const [phase,  setPhase]  = useState("idle");
  const [predResult,  setPredResult]  = useState(null);
  const [learnReport, setLearnReport] = useState(null);
  const [learnMeta,   setLearnMeta]   = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState(null);

  /* load (migration-safe merge for engine + cross-match fields) */
  useEffect(()=>{
    (async()=>{
      try {
        const s = await window.storage.get(STORAGE_KEY);
        if (s?.value) {
          const parsed = JSON.parse(s.value);
          setKb({
            ...INIT_STATE, ...parsed,
            mFrequencies: {...INIT_STATE.mFrequencies, ...(parsed.mFrequencies||{})},
            mPatterns: parsed.mPatterns || {},
            mHistory: parsed.mHistory || [],
            crossMatches: parsed.crossMatches || [],
          });
        }
      } catch {}
      setLoaded(true);
    })();
  },[]);

  const persist=useCallback(async st=>{
    try { await window.storage.set(STORAGE_KEY,JSON.stringify(st)); } catch {}
  },[]);

  /* ── PREDICT — classification happens here: M in input → M-engine, else standard ── */
  const handlePredict=useCallback(async()=>{
    if(!syms.every(Boolean)||loading) return;
    setError(null); setPredResult(null); setLearnReport(null); setLearnMeta(null);

    if (hasM(syms)) {
      setPhase("predicted");
      setPredResult(predictMSequence(syms, kb));
      return;
    }

    setLoading(true); setPhase("predicting");
    try {
      const contextKey = syms.join("");
      const r=await callClaude(
        buildPredPrompt(kb, contextKey),
        `INPUT\n1st symbol=[${syms[0]}]\n2nd symbol=[${syms[1]}]\n3rd symbol=[${syms[2]}]\n\nAnalyze and predict the 4th symbol.`
      );
      setPredResult({...r, _type:"standard", _contextKey:contextKey}); setPhase("predicted");
    } catch { setError("Analysis failed — please try again."); setPhase("idle"); }
    finally { setLoading(false); }
  },[syms,kb,loading]);

  /* ── LEARN — updates ONLY the matching engine's weights, then runs the
       mandatory opposite-sequence cross-check (memory-only, never reclassifies) ── */
  const handleLearn=useCallback(async()=>{
    if(!predResult||!actual) return;
    setError(null);

    if (predResult._type === "m-sequence") {
      setPhase("learning");
      const ctx = predResult._ctx, flip = predResult._flip;
      const correct = flip!==null && flip===actual;
      const before = kb.mWeight;
      const newWeight = Math.min(MAX_W, Math.max(MIN_W, before + (correct?REWARD:-PENALTY)));
      const newMFreq = {...kb.mFrequencies, [actual]: (kb.mFrequencies[actual]||0)+1};
      const contextKey = ctx?.prevSym ?? "∅";
      const patternKey = `${contextKey}→M→${actual}`;
      const newMPatterns = {...kb.mPatterns, [patternKey]: (kb.mPatterns[patternKey]||0)+1};
      const newMHistory = [{prevSym:ctx?.prevSym??null, flip, actual, correct, ts:Date.now()}, ...kb.mHistory].slice(0,50);

      // mandatory opposite-sequence check — always runs, never skipped
      const shadow = shadowCandidate(kb, "m-sequence");
      const crossEntry = (shadow && shadow===actual)
        ? { ts:Date.now(), primaryType:"m-sequence", contextKey, actual, shadowType:"standard", shadowCandidate:shadow }
        : null;
      const newCrossMatches = crossEntry ? [crossEntry, ...(kb.crossMatches||[])].slice(0,30) : (kb.crossMatches||[]);

      const newState = {
        ...kb, mWeight:newWeight, mFrequencies:newMFreq, mPatterns:newMPatterns,
        mHistory:newMHistory, mTotalPred:kb.mTotalPred+1, mCorrectPred:kb.mCorrectPred+(correct?1:0),
        lastAnswer:actual, crossMatches:newCrossMatches,
      };

      const report = {
        result: correct?"CORRECT":"INCORRECT",
        classification: "M-SEQUENCE (boundary marker detected in input)",
        patternsRewarded: correct?[`M-flip rule (${before.toFixed(2)}→${newWeight.toFixed(2)})`]:["None"],
        patternsPenalized: !correct?[`M-flip rule (${before.toFixed(2)}→${newWeight.toFixed(2)})`]:["None"],
        crossCheck: crossEntry
          ? `MATCH — the standard engine's frequency bias also pointed to ${shadow}, which matched the actual outcome. Noted for future M-rounds with this same preceding answer, without touching the M-engine's own weight update.`
          : `NO MATCH — the standard engine's frequency bias (${shadow ?? "n/a"}) did not align with the actual outcome this round.`,
        updatedFrequencies: {U:kb.frequencies.U, D:kb.frequencies.D, M:kb.frequencies.M},
        newSequenceStored: "YES",
        knowledgeBaseUpdated: "YES",
        whatChanged: `M-sequence engine: the answer preceding M was ${ctx?.prevSym??"unknown (start)"}, so the locked flip rule predicted ${flip??"—"}. The actual outcome was ${actual}, making this ${correct?"CORRECT":"INCORRECT"}. The flip-rule reliability weight moved from ${before.toFixed(2)} to ${newWeight.toFixed(2)}, and pattern "${patternKey}" was stored in the M-sequence knowledge base — kept fully separate from the standard sequence/motif weights.`,
      };
      setKb(newState); persist(newState);
      setLearnReport(report);
      setLearnMeta({partial:patternKey, newSeqStored:true, type:"m-sequence"});
      setPhase("learned");
      return;
    }

    setLoading(true); setPhase("learning");
    try {
      const {newState,rewarded,penalized,newFreq,newSeqStored,partial} =
        applyWeights(kb, predResult.prediction, actual, predResult.rewardKeys||[], predResult.penaltyKeys||[], syms);

      // mandatory opposite-sequence check — always runs, never skipped
      const shadow = shadowCandidate(kb, "standard");
      const contextKey = syms.join("");
      const crossEntry = (shadow && shadow===actual)
        ? { ts:Date.now(), primaryType:"standard", contextKey, actual, shadowType:"m-sequence", shadowCandidate:shadow }
        : null;
      const newCrossMatches = crossEntry ? [crossEntry, ...(kb.crossMatches||[])].slice(0,30) : (kb.crossMatches||[]);

      const finalState = {...newState, lastAnswer:actual, crossMatches:newCrossMatches};

      const report = await callClaude(
        buildLearnPrompt(kb, syms, predResult.prediction, actual, rewarded, penalized, newSeqStored, newFreq),
        `Generate the learning update report for this outcome.`
      );
      report.classification = "STANDARD SEQUENCE (no M in input window)";
      report.crossCheck = crossEntry
        ? `MATCH — the M-engine's opposite-of-previous shadow rule (flip of lastAnswer=${kb.lastAnswer??"∅"}) also predicted ${shadow}, which matched the actual outcome. Noted for future inputs like [${contextKey}], without touching the standard engine's own weight update.`
        : `NO MATCH — the M-engine's shadow flip (${shadow ?? "n/a"}) did not align with the actual outcome this round.`;

      setKb(finalState); persist(finalState);
      setLearnReport(report);
      setLearnMeta({partial, newSeqStored, type:"standard"});
      setPhase("learned");
    } catch {
      setError("Learning update failed — please try again."); setPhase("predicted");
    }
    finally { setLoading(false); }
  },[predResult,actual,kb,syms,persist]);

  const resetSession=()=>{
    setSyms([null,null,null]); setActual(null); setPredResult(null);
    setLearnReport(null); setLearnMeta(null); setError(null); setPhase("idle");
  };
  const hardReset=async()=>{
    setKb(INIT_STATE); resetSession();
    try { await window.storage.delete(STORAGE_KEY); } catch {}
  };

  const canPredict = syms.every(Boolean)&&!loading&&(phase==="idle");
  const canLearn   = !!predResult&&!!actual&&!loading&&phase==="predicted";
  const liveIsM    = syms.some(Boolean) && hasM(syms);

  const card={background:CARD,border:`1px solid ${BORDER}`,borderRadius:11};
  const lbl={color:"#253448",fontSize:9.5,fontFamily:"'Fira Code',monospace",
    letterSpacing:"0.08em",textTransform:"uppercase",marginBottom:5};

  /* ═══ RENDER ═══ */
  return (
    <div style={{minHeight:"100vh",background:BG,color:"#e2e8f0",
      fontFamily:"'DM Sans',system-ui,sans-serif",padding:"22px 14px 60px"}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&family=Fira+Code:wght@400;500;600;700&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:4px} ::-webkit-scrollbar-thumb{background:#1a2d45;border-radius:2px}
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
        @keyframes blink{0%,100%{opacity:.2}50%{opacity:.8}}
        .btn{transition:all .14s;cursor:pointer}
        .btn:not(:disabled):hover{filter:brightness(1.14);transform:translateY(-1px)}
        .btn:active{transform:scale(.97)}
        .sk{transition:all .12s;cursor:pointer}
        .sk:hover{filter:brightness(1.2);transform:translateY(-1px)}
        .sk:active{transform:scale(.94)}
      `}</style>

      <div style={{maxWidth:840,margin:"0 auto"}}>

        {/* HEADER */}
        <div style={{marginBottom:20,display:"flex",alignItems:"flex-start",
          justifyContent:"space-between",flexWrap:"wrap",gap:10}}>
          <div>
            <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:4}}>
              <div style={{display:"flex",gap:3}}>
                {[UP_C,MID_C,DOWN_C].map(c=>(
                  <div key={c} style={{width:3,height:24,background:c,borderRadius:2}}/>
                ))}
              </div>
              <h1 style={{fontFamily:"'Syne',sans-serif",fontSize:21,fontWeight:800,
                letterSpacing:"-0.03em",color:"#f1f5f9"}}>
                Self-Learning Prediction Engine
              </h1>
            </div>
            <p style={{color:"#1e3048",fontSize:11,fontFamily:"'Fira Code',monospace",paddingLeft:18}}>
              Two isolated weight sets · mandatory opposite-sequence cross-check · persists across sessions
            </p>
          </div>
          <div style={{display:"flex",gap:7}}>
            {phase!=="idle"&&(
              <button className="btn" onClick={resetSession}
                style={{background:"transparent",border:`1px solid ${BORDER}`,borderRadius:7,
                  padding:"5px 12px",color:"#334155",fontSize:11,cursor:"pointer"}}>
                ↺ New Prediction
              </button>
            )}
            <button className="btn" onClick={hardReset}
              style={{background:"transparent",border:"1px solid rgba(255,77,114,.2)",borderRadius:7,
                padding:"5px 12px",color:DOWN_C,fontSize:11,cursor:"pointer",opacity:.55}}>
              Reset KB
            </button>
          </div>
        </div>

        {/* TWO-COLUMN */}
        <div style={{display:"grid",gridTemplateColumns:"1fr 290px",gap:12,alignItems:"start"}}>

          {/* LEFT */}
          <div>

            {/* INPUT CARD */}
            <div style={{...card,padding:"18px 16px",marginBottom:12}}>
              <div style={{background:"#070d19",border:`1px solid ${BORDER}`,borderRadius:8,
                padding:"10px 13px",marginBottom:14,fontFamily:"'Fira Code',monospace"}}>
                <p style={{color:"#1e3048",fontSize:9,letterSpacing:"0.08em",
                  textTransform:"uppercase",marginBottom:6}}>INPUT</p>
                <p style={{color:"#253448",fontSize:11,marginBottom:3}}>Previous Sequence:</p>
                <p style={{color:"#334155",fontSize:13,marginBottom:8,letterSpacing:"0.05em"}}>
                  {[0,1,2].map(i=>(
                    <span key={i} style={{color:syms[i]?symColor(syms[i]):"#1e3048"}}>
                      [{syms[i]||" "}]
                    </span>
                  ))}
                </p>
                <p style={{color:"#253448",fontSize:11,marginBottom:3}}>Predicted Symbol:</p>
                <p style={{color:predResult?symColor(predResult.prediction):"#1e3048",
                  fontSize:13,marginBottom:8,letterSpacing:"0.05em"}}>
                  [{predResult?.prediction||" "}]
                </p>
                <p style={{color:"#253448",fontSize:11,marginBottom:3}}>Actual Outcome:</p>
                <p style={{color:actual?symColor(actual):"#1e3048",
                  fontSize:13,letterSpacing:"0.05em"}}>
                  [{actual||" "}]
                </p>
              </div>

              <p style={lbl}>Select Symbols</p>

              {syms.some(Boolean) && (
                <div style={{display:"flex",alignItems:"center",gap:7,marginBottom:10}}>
                  <span style={{
                    width:6,height:6,borderRadius:"50%",
                    background: liveIsM ? MID_C : BLUE,
                    boxShadow:`0 0 6px ${liveIsM?MID_C:BLUE}`}}/>
                  <span style={{
                    color: liveIsM ? MID_C : BLUE,
                    fontFamily:"'Fira Code',monospace", fontSize:10.5, fontWeight:700,
                    letterSpacing:"0.05em", textTransform:"uppercase"}}>
                    {liveIsM ? "M detected → new sequence boundary, flip-rule engine" : "Standard sequence → pattern-matching engine"}
                  </span>
                </div>
              )}

              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:8,marginBottom:12}}>
                {[0,1,2].map(i=>{
                  const s=syms[i],c=s?symColor(s):"#1e3048",rgb=s?symRgb(s):"30,48,72";
                  const locked=phase!=="idle";
                  return (
                    <div key={i} style={{background:"#060d1a",
                      border:`2px solid ${s?c+"55":BORDER}`,borderRadius:9,
                      padding:"10px 8px 9px",
                      boxShadow:s?`0 0 14px rgba(${rgb},.12)`:"none",
                      transition:"all .3s",opacity:locked?.6:1}}>
                      <p style={{color:"#1a2a38",fontSize:8.5,fontFamily:"'Fira Code',monospace",
                        letterSpacing:"0.08em",textTransform:"uppercase",
                        textAlign:"center",marginBottom:5}}>Pos {i+1}</p>
                      <div style={{height:40,display:"flex",alignItems:"center",
                        justifyContent:"center",marginBottom:s==="M"?2:7}}>
                        {s
                          ? <span style={{fontFamily:"'Syne',sans-serif",fontSize:34,fontWeight:800,
                              color:c,lineHeight:1,textShadow:`0 0 22px ${c}55`,
                              animation:"fadeUp .2s ease"}}>{s}</span>
                          : <span style={{fontFamily:"'Fira Code',monospace",fontSize:26,color:"#1e3048",
                              animation:"blink 1.5s ease infinite"}}>_</span>
                        }
                      </div>
                      {s==="M" && (
                        <p style={{color:MID_C,fontSize:8,fontFamily:"'Fira Code',monospace",
                          textAlign:"center",marginBottom:5,letterSpacing:"0.04em"}}>
                          ⟲ boundary
                        </p>
                      )}
                      <div style={{display:"flex",gap:4}}>
                        {["U","D","M"].map(sym=>(
                          <SymKey key={sym} sym={sym} active={syms[i]===sym}
                            disabled={locked}
                            onClick={()=>setSyms(p=>{const n=[...p];n[i]=sym;return n;})}/>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>

              {phase==="idle"&&(
                <button className="btn" onClick={handlePredict} disabled={!canPredict} style={{
                  width:"100%",padding:"13px 0",borderRadius:8,
                  fontSize:14,fontFamily:"'Syne',sans-serif",fontWeight:800,
                  letterSpacing:"0.07em",textTransform:"uppercase",
                  cursor:canPredict?"pointer":"not-allowed",
                  background:canPredict
                    ? (liveIsM ? "linear-gradient(135deg,#2a1d4d,#0a2248)" : "linear-gradient(135deg,#0d3d2f,#0a2248)")
                    : `#080f1c`,
                  border:canPredict
                    ? `1px solid ${liveIsM?"rgba(167,139,250,.32)":"rgba(0,229,160,.28)"}`
                    : "1px solid #172435",
                  color:canPredict?"#e2e8f0":"#1e3048",
                  boxShadow:canPredict
                    ? `0 0 20px ${liveIsM?"rgba(167,139,250,.12)":"rgba(0,229,160,.1)"}`
                    : "none",
                }}>
                  {loading
                    ? <span style={{display:"inline-flex",alignItems:"center",gap:10}}>
                        <span style={{width:13,height:13,border:"2px solid rgba(0,229,160,.3)",
                          borderTopColor:UP_C,borderRadius:"50%",
                          animation:"spin .6s linear infinite",display:"inline-block"}}/>
                        Analyzing…
                      </span>
                    : liveIsM ? "◈  Apply Flip Rule" : "◈  Analyze & Predict"}
                </button>
              )}
              {error&&<p style={{color:DOWN_C,fontSize:12,textAlign:"center",
                fontFamily:"'Fira Code',monospace",marginTop:8}}>{error}</p>}
            </div>

            {/* ═══ WHAT I GOT ═══ */}
            {phase==="predicted"&&(
              <div style={{animation:"fadeUp .35s ease",marginBottom:12}}>

                <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:8}}>
                  <span style={{width:6,height:6,borderRadius:"50%",
                    background:predResult._type==="m-sequence"?MID_C:BLUE}}/>
                  <span style={{color:predResult._type==="m-sequence"?MID_C:BLUE,
                    fontFamily:"'Fira Code',monospace",fontSize:10,fontWeight:700,
                    letterSpacing:"0.05em",textTransform:"uppercase"}}>
                    {predResult._type==="m-sequence" ? "M-Engine — opposite-of-previous rule" : "Standard Engine — pattern matching"}
                  </span>
                </div>

                <div style={{
                  display:"flex",alignItems:"center",gap:0,
                  marginBottom:10,borderRadius:10,overflow:"hidden",
                  border:`1px solid ${BORDER}`,
                }}>
                  <div style={{
                    flex:1,padding:"10px 14px",background:"#080f1c",
                    borderRight:`1px solid ${BORDER}`,
                  }}>
                    <p style={{color:"#1e3048",fontSize:9,fontFamily:"'Fira Code',monospace",
                      letterSpacing:"0.08em",textTransform:"uppercase",marginBottom:3}}>
                      Engine Predicted
                    </p>
                    <span style={{
                      fontFamily:"'Syne',sans-serif",fontSize:30,fontWeight:800,lineHeight:1,
                      color:symColor(predResult?.prediction),
                      textShadow:`0 0 20px ${symColor(predResult?.prediction)}60`,
                    }}>{predResult?.prediction||"—"}</span>
                  </div>
                  <div style={{
                    padding:"0 12px",background:"#080f1c",
                    color:"#1e3048",fontFamily:"'Fira Code',monospace",fontSize:18,
                  }}>vs</div>
                  <div style={{
                    flex:1,padding:"10px 14px",
                    background: actual
                      ? `rgba(${symRgb(actual)},.08)`
                      : "#080f1c",
                    borderLeft:`1px solid ${BORDER}`,
                    transition:"background .3s",
                  }}>
                    <p style={{color:"#1e3048",fontSize:9,fontFamily:"'Fira Code',monospace",
                      letterSpacing:"0.08em",textTransform:"uppercase",marginBottom:3}}>
                      What I Got
                    </p>
                    {actual
                      ? <span style={{
                          fontFamily:"'Syne',sans-serif",fontSize:30,fontWeight:800,lineHeight:1,
                          color:symColor(actual),
                          textShadow:`0 0 20px ${symColor(actual)}60`,
                          animation:"fadeUp .2s ease",
                        }}>{actual}</span>
                      : <span style={{fontFamily:"'Fira Code',monospace",fontSize:24,
                          color:"#1e3048",animation:"blink 1.5s ease infinite"}}>_</span>
                    }
                  </div>
                  {actual&&(
                    <div style={{
                      padding:"0 14px",
                      background: actual===predResult?.prediction
                        ? "rgba(0,229,160,.1)"
                        : "rgba(255,77,114,.1)",
                      borderLeft:`1px solid ${BORDER}`,
                      display:"flex",flexDirection:"column",alignItems:"center",
                      justifyContent:"center",gap:2,alignSelf:"stretch",
                    }}>
                      <span style={{
                        fontSize:20,
                        color: actual===predResult?.prediction ? UP_C : DOWN_C,
                      }}>
                        {actual===predResult?.prediction ? "✓" : "✗"}
                      </span>
                      <span style={{
                        color: actual===predResult?.prediction ? UP_C : DOWN_C,
                        fontFamily:"'Fira Code',monospace",fontSize:8,fontWeight:700,
                        letterSpacing:"0.06em",textTransform:"uppercase",
                      }}>
                        {actual===predResult?.prediction ? "MATCH" : "MISS"}
                      </span>
                    </div>
                  )}
                </div>

                <div style={{
                  background:"#0a1220",
                  border:`2px solid ${actual?symColor(actual)+"50":MID_C+"35"}`,
                  borderRadius:11,overflow:"hidden",
                  boxShadow: actual ? `0 0 24px rgba(${symRgb(actual)},.1)` : "none",
                  transition:"all .3s",
                }}>
                  <div style={{
                    background:`linear-gradient(90deg,rgba(167,139,250,.14) 0%,transparent 60%)`,
                    borderBottom:`1px solid rgba(167,139,250,.2)`,
                    padding:"10px 18px",
                    display:"flex",alignItems:"center",gap:10,
                  }}>
                    <span style={{
                      width:8,height:8,borderRadius:"50%",
                      background:MID_C,boxShadow:`0 0 6px ${MID_C}`,
                      display:"inline-block",
                    }}/>
                    <span style={{color:MID_C,fontSize:12,fontFamily:"'Syne',sans-serif",
                      fontWeight:700,letterSpacing:"0.04em"}}>
                      What I Got
                    </span>
                    <span style={{color:"#1e3048",fontSize:10,fontFamily:"'Fira Code',monospace",
                      marginLeft:4}}>
                      — enter the actual outcome you observed
                    </span>
                  </div>

                  <div style={{padding:"18px 18px 16px"}}>
                    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:10,marginBottom:16}}>
                      {["U","D","M"].map(sym=>{
                        const c=symColor(sym), r=symRgb(sym), active=actual===sym;
                        return (
                          <button key={sym} className="sk" onClick={()=>setActual(sym)} style={{
                            padding:"18px 0",borderRadius:10,cursor:"pointer",
                            fontFamily:"'Syne',sans-serif",fontSize:32,fontWeight:800,
                            background:active?`rgba(${r},.16)`:"rgba(255,255,255,.02)",
                            border:`2px solid ${active?c+"70":BORDER}`,
                            color:active?c:"#253448",
                            boxShadow:active?`0 0 20px rgba(${r},.25), inset 0 0 12px rgba(${r},.05)`:"none",
                            transition:"all .15s",
                            display:"flex",flexDirection:"column",alignItems:"center",gap:4,
                          }}>
                            <span>{sym}</span>
                            <span style={{fontSize:10,fontFamily:"'Fira Code',monospace",
                              fontWeight:600,letterSpacing:"0.05em",
                              color:active?c+"99":"#1a2a38"}}>
                              {sym==="U"?"UP":sym==="D"?"DOWN":"MID"}
                            </span>
                          </button>
                        );
                      })}
                    </div>

                    <button className="btn" onClick={handleLearn} disabled={!canLearn} style={{
                      width:"100%",padding:"14px 0",borderRadius:9,
                      fontSize:14,fontFamily:"'Syne',sans-serif",fontWeight:800,
                      letterSpacing:"0.07em",textTransform:"uppercase",
                      cursor:canLearn?"pointer":"not-allowed",
                      background:canLearn
                        ? `linear-gradient(135deg, rgba(${symRgb(actual||"M")},.22) 0%, #0a1a30 100%)`
                        : "#080f1c",
                      border:canLearn
                        ? `1px solid rgba(${symRgb(actual||"M")},.45)`
                        : "1px solid #172435",
                      color:canLearn?"#e2e8f0":"#1e3048",
                      boxShadow:canLearn?`0 0 18px rgba(${symRgb(actual||"M")},.14)`:"none",
                      transition:"all .25s",
                    }}>
                      {loading
                        ? <span style={{display:"inline-flex",alignItems:"center",gap:10}}>
                            <span style={{width:13,height:13,
                              border:`2px solid rgba(${symRgb(actual||"M")},.3)`,
                              borderTopColor:symColor(actual||"M"),borderRadius:"50%",
                              animation:"spin .6s linear infinite",display:"inline-block"}}/>
                            Updating Knowledge Base…
                          </span>
                        : actual
                          ? `◆ Record ${actual} & Learn`
                          : "◆ Select a symbol above first"
                      }
                    </button>

                    {error&&<p style={{color:DOWN_C,fontSize:11,textAlign:"center",
                      fontFamily:"'Fira Code',monospace",marginTop:8}}>{error}</p>}
                  </div>
                </div>
              </div>
            )}

            {learnReport&&(
              <div style={{animation:"fadeUp .4s ease"}}>
                <LearnReport report={learnReport} partial={learnMeta?.partial} engineType={learnMeta?.type}/>
              </div>
            )}

            {predResult&&(
              <div style={{animation:"fadeUp .4s ease"}}>
                <PredOutput result={predResult}/>
              </div>
            )}

            {/* DATABASE OVERLAY — standard engine only */}
            {predResult && predResult._type==="standard" && (
              <div style={{...card,padding:"12px 14px"}}>
                <p style={lbl}>Database · Match Overlay</p>
                <div style={{display:"flex",flexWrap:"wrap",gap:6,marginBottom:8}}>
                  {kb.sequences.map(seq=>{
                    const type=predResult.matchingSequences?.includes(seq)?"exact"
                      :predResult.nearMatches?.some(m=>m.startsWith(seq))?"near"
                      :predResult.rejectedSequences?.some(r=>(r+" ").startsWith(seq+" ")||r===seq)?"rejected"
                      :"neutral";
                    return <SeqChip key={seq} seq={seq} type={type} weight={kb.seqWeights[seq]||1}/>;
                  })}
                </div>
                <div style={{display:"flex",gap:12,flexWrap:"wrap"}}>
                  {[{c:UP_C,l:"Exact match"},{c:GOLD,l:"Near match"},{c:"#1e3048",l:"Rejected"}].map(({c,l})=>(
                    <div key={l} style={{display:"flex",alignItems:"center",gap:4}}>
                      <div style={{width:7,height:7,borderRadius:2,background:c}}/>
                      <span style={{color:"#1e3048",fontSize:9.5,fontFamily:"'Fira Code',monospace"}}>{l}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* RIGHT — KB */}
          {loaded&&<KBPanel state={kb}/>}
        </div>

        <div style={{marginTop:22,textAlign:"center",color:"#0f1e2e",
          fontSize:10,fontFamily:"'Fira Code',monospace"}}>
          Self-Learning Sequence Engine · isolated weights · mandatory cross-check · storage-persistent
        </div>
      </div>
    </div>
  );
}
