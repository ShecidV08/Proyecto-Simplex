/* SimplexLab
 * Implementación didáctica del Método Simplex:
 * - Forma estándar: restricciones como igualdades con holgura, exceso y artificiales.
 * - Criterio primal: columna más negativa en fila Z; fila por cociente positivo mínimo.
 * - Fase I: maximiza -Σ artificiales para detectar infactibilidad.
 * - Fase II: restituye la función objetivo original.
 */

const EPS = 1e-9;
const MAX_ITER = 80;

const examples = [
  {
    id: "max-le",
    title: "Ejemplo 1 · Maximización con ≤",
    description: "Max Z = 3x₁ + 5x₂. Óptimo esperado: x₁=2, x₂=6, Z=36.",
    model: {
      sense: "max",
      method: "primal",
      c: [3, 5],
      constraints: [
        { a: [1, 0], sign: "<=", b: 4 },
        { a: [0, 2], sign: "<=", b: 12 },
        { a: [3, 2], sign: "<=", b: 18 },
      ],
    },
  },
  {
    id: "min-ge",
    title: "Ejemplo 2 · Minimización con ≥",
    description: "Min Z = 4x₁ + x₂ con restricciones ≥. Requiere Dos Fases o Dual.",
    model: {
      sense: "min",
      method: "two-phase",
      c: [4, 1],
      constraints: [
        { a: [3, 1], sign: ">=", b: 3 },
        { a: [4, 3], sign: ">=", b: 6 },
      ],
    },
  },
  {
    id: "artificial",
    title: "Ejemplo 3 · Igualdad y artificiales",
    description: "Incluye una igualdad y una restricción ≥. Fase I y Fase II.",
    model: {
      sense: "max",
      method: "two-phase",
      c: [1, 1],
      constraints: [
        { a: [1, 1], sign: "=", b: 4 },
        { a: [1, 2], sign: ">=", b: 6 },
      ],
    },
  },
  {
    id: "unbounded",
    title: "Ejemplo 4 · No acotamiento",
    description: "Max Z = x₁ + x₂; la región permite crecer indefinidamente.",
    model: {
      sense: "max",
      method: "two-phase",
      c: [1, 1],
      constraints: [{ a: [1, -1], sign: "<=", b: 2 }],
    },
  },
  {
    id: "infeasible",
    title: "Ejemplo 5 · Infactibilidad",
    description: "Restricciones incompatibles: x₁+x₂ ≤ 1 y x₁+x₂ ≥ 3.",
    model: {
      sense: "max",
      method: "two-phase",
      c: [1, 1],
      constraints: [
        { a: [1, 1], sign: "<=", b: 1 },
        { a: [1, 1], sign: ">=", b: 3 },
      ],
    },
  },
  {
    id: "decimals",
    title: "Ejemplo 6 · Coeficientes decimales",
    description: "Max Z = 2.5x₁ + 1.75x₂. Demuestra entrada con números decimales.",
    model: {
      sense: "max",
      method: "two-phase",
      c: [2.5, 1.75],
      constraints: [
        { a: [1.5, 1], sign: "<=", b: 6 },
        { a: [0.5, 2], sign: "<=", b: 8 },
        { a: [1, 0], sign: "<=", b: 5 },
      ],
    },
  },
];

function cleanNumber(value) {
  if (Math.abs(value) < EPS) return 0;
  return Number.parseFloat(value.toFixed(8));
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "—";
  const clean = cleanNumber(value);
  return Math.abs(clean - Math.round(clean)) < 1e-8
    ? String(Math.round(clean))
    : clean.toLocaleString("es-BO", { maximumFractionDigits: 6 });
}

function cloneTableau(tableau) {
  return tableau.map((row) => row.slice());
}

function normalizeSign(sign) {
  return sign === "≤" ? "<=" : sign === "≥" ? ">=" : sign;
}

function flipSign(sign) {
  if (sign === "<=") return ">=";
  if (sign === ">=") return "<=";
  return "=";
}

function validateModel(model) {
  const errors = [];
  const n = model.c.length;
  if (n < 1 || n > 10) errors.push("El número de variables debe estar entre 1 y 10.");
  if (model.constraints.length < 1 || model.constraints.length > 10) {
    errors.push("El número de restricciones debe estar entre 1 y 10.");
  }
  model.c.forEach((value, index) => {
    if (!Number.isFinite(value)) errors.push(`Coeficiente inválido en Z para x${index + 1}.`);
  });
  model.constraints.forEach((constraint, rowIndex) => {
    if (constraint.a.length !== n) errors.push(`La restricción ${rowIndex + 1} no tiene ${n} coeficientes.`);
    constraint.a.forEach((value, colIndex) => {
      if (!Number.isFinite(value)) {
        errors.push(`Coeficiente inválido en restricción ${rowIndex + 1}, x${colIndex + 1}.`);
      }
    });
    if (!Number.isFinite(constraint.b)) errors.push(`Lado derecho inválido en restricción ${rowIndex + 1}.`);
  });
  return errors;
}

function standardizeInput(model) {
  return {
    sense: model.sense,
    c: model.c.slice(),
    constraints: model.constraints.map((constraint) => {
      let sign = normalizeSign(constraint.sign);
      let a = constraint.a.slice();
      let b = constraint.b;
      if (b < -EPS) {
        a = a.map((value) => -value);
        b = -b;
        sign = flipSign(sign);
      }
      return { a, sign, b };
    }),
  };
}

function buildGeneralTableau(model) {
  const cleanModel = standardizeInput(model);
  const originalN = cleanModel.c.length;
  const varNames = Array.from({ length: originalN }, (_, index) => `x${index + 1}`);
  const varTypes = Array.from({ length: originalN }, () => "decision");
  const artificial = new Set();
  const basis = [];
  const rows = [];

  function addVariable(name, type) {
    varNames.push(name);
    varTypes.push(type);
    rows.forEach((row) => row.splice(row.length - 1, 0, 0));
    return varNames.length - 1;
  }

  cleanModel.constraints.forEach((constraint, rowIndex) => {
    const row = Array(varNames.length + 1).fill(0);
    constraint.a.forEach((value, colIndex) => {
      row[colIndex] = value;
    });
    row[row.length - 1] = constraint.b;
    rows.push(row);

    if (constraint.sign === "<=") {
      const slackIndex = addVariable(`s${rowIndex + 1}`, "slack");
      row[slackIndex] = 1;
      basis.push(slackIndex);
    } else if (constraint.sign === ">=") {
      const surplusIndex = addVariable(`e${rowIndex + 1}`, "surplus");
      row[surplusIndex] = -1;
      const artificialIndex = addVariable(`a${rowIndex + 1}`, "artificial");
      row[artificialIndex] = 1;
      artificial.add(artificialIndex);
      basis.push(artificialIndex);
    } else {
      const artificialIndex = addVariable(`a${rowIndex + 1}`, "artificial");
      row[artificialIndex] = 1;
      artificial.add(artificialIndex);
      basis.push(artificialIndex);
    }
  });

  const cols = varNames.length;
  const tableau = rows.map((row) => {
    while (row.length < cols + 1) row.splice(row.length - 1, 0, 0);
    return row;
  });

  return {
    tableau,
    basis,
    varNames,
    varTypes,
    artificial,
    originalN,
    cleanModel,
  };
}

function setObjectiveRow(state, costs) {
  const cols = state.varNames.length;
  const objective = Array(cols + 1).fill(0);
  for (let j = 0; j < cols; j += 1) objective[j] = -(costs[j] || 0);
  objective[cols] = 0;

  state.basis.forEach((basicIndex, rowIndex) => {
    const cB = costs[basicIndex] || 0;
    if (Math.abs(cB) < EPS) return;
    for (let j = 0; j <= cols; j += 1) {
      objective[j] += cB * state.tableau[rowIndex][j];
    }
  });

  state.tableau = state.tableau.slice(0, state.basis.length).concat([objective]);
}

function pivot(state, rowIndex, colIndex) {
  const cols = state.varNames.length;
  const pivotValue = state.tableau[rowIndex][colIndex];
  for (let j = 0; j <= cols; j += 1) {
    state.tableau[rowIndex][j] /= pivotValue;
  }
  for (let i = 0; i < state.tableau.length; i += 1) {
    if (i === rowIndex) continue;
    const factor = state.tableau[i][colIndex];
    if (Math.abs(factor) < EPS) continue;
    for (let j = 0; j <= cols; j += 1) {
      state.tableau[i][j] -= factor * state.tableau[rowIndex][j];
    }
  }
  state.basis[rowIndex] = colIndex;
}

function snapshot(state, meta) {
  return {
    phase: meta.phase,
    title: meta.title,
    summary: meta.summary,
    entering: meta.entering ?? null,
    leaving: meta.leaving ?? null,
    pivotValue: meta.pivotValue ?? null,
    ratios: meta.ratios || [],
    tableau: cloneTableau(state.tableau),
    basis: state.basis.slice(),
    varNames: state.varNames.slice(),
  };
}

function runPrimalSimplex(state, phaseLabel) {
  const iterations = [];
  const cols = state.varNames.length;
  const objectiveRowIndex = state.basis.length;
  let count = 0;

  while (count < MAX_ITER) {
    const objective = state.tableau[objectiveRowIndex];
    let entering = -1;
    let mostNegative = -EPS;
    for (let j = 0; j < cols; j += 1) {
      if (objective[j] < mostNegative) {
        mostNegative = objective[j];
        entering = j;
      }
    }

    if (entering === -1) {
      iterations.push(
        snapshot(state, {
          phase: phaseLabel,
          title: "Prueba de optimalidad satisfecha",
          summary: "La fila Z ya no contiene coeficientes negativos; el tablero es óptimo para esta fase.",
        }),
      );
      return { status: "optimal", iterations };
    }

    const ratios = [];
    let leaving = -1;
    let bestRatio = Infinity;
    for (let i = 0; i < state.basis.length; i += 1) {
      const coefficient = state.tableau[i][entering];
      const rhs = state.tableau[i][cols];
      const ratio = coefficient > EPS ? rhs / coefficient : Infinity;
      ratios.push(Number.isFinite(ratio) ? ratio : null);
      if (coefficient > EPS && ratio >= -EPS && ratio < bestRatio - EPS) {
        bestRatio = ratio;
        leaving = i;
      }
    }

    if (leaving === -1) {
      iterations.push(
        snapshot(state, {
          phase: phaseLabel,
          title: "No acotamiento detectado",
          entering,
          ratios,
          summary: `La variable ${state.varNames[entering]} puede entrar, pero no existe cociente positivo mínimo.`,
        }),
      );
      return { status: "unbounded", iterations, entering };
    }

    iterations.push(
      snapshot(state, {
        phase: phaseLabel,
        title: `Pivote ${count + 1}`,
        entering,
        leaving,
        pivotValue: state.tableau[leaving][entering],
        ratios,
        summary: `${state.varNames[entering]} entra por ser la columna más negativa; ${state.varNames[state.basis[leaving]]} sale por el cociente positivo mínimo.`,
      }),
    );

    pivot(state, leaving, entering);
    count += 1;
  }

  return {
    status: "iteration-limit",
    iterations,
  };
}

function removeArtificialColumns(state) {
  const cols = state.varNames.length;
  let rowIndex = 0;
  while (rowIndex < state.basis.length) {
    const basic = state.basis[rowIndex];
    if (!state.artificial.has(basic)) {
      rowIndex += 1;
      continue;
    }

    let replacement = -1;
    for (let j = 0; j < cols; j += 1) {
      if (!state.artificial.has(j) && Math.abs(state.tableau[rowIndex][j]) > EPS) {
        replacement = j;
        break;
      }
    }

    if (replacement !== -1) {
      pivot(state, rowIndex, replacement);
      rowIndex += 1;
    } else {
      state.tableau.splice(rowIndex, 1);
      state.basis.splice(rowIndex, 1);
    }
  }

  const keep = [];
  const remap = new Map();
  for (let j = 0; j < state.varNames.length; j += 1) {
    if (!state.artificial.has(j)) {
      remap.set(j, keep.length);
      keep.push(j);
    }
  }

  state.tableau = state.tableau.slice(0, state.basis.length).map((row) => {
    const next = keep.map((oldIndex) => row[oldIndex]);
    next.push(row[row.length - 1]);
    return next;
  });
  state.basis = state.basis.map((oldIndex) => remap.get(oldIndex));
  state.varNames = keep.map((oldIndex) => state.varNames[oldIndex]);
  state.varTypes = keep.map((oldIndex) => state.varTypes[oldIndex]);
  state.artificial = new Set();
}

function currentValues(state) {
  const cols = state.varNames.length;
  const values = Array(cols).fill(0);
  state.basis.forEach((basicIndex, rowIndex) => {
    values[basicIndex] = state.tableau[rowIndex][cols];
  });
  return values;
}

function buildOriginalCosts(model, totalVars, originalN) {
  const costs = Array(totalVars).fill(0);
  const multiplier = model.sense === "max" ? 1 : -1;
  for (let j = 0; j < originalN; j += 1) costs[j] = multiplier * model.c[j];
  return costs;
}

function analyzeFinalState(state, model, phaseStatus) {
  const values = currentValues(state).map(cleanNumber);
  const originalValues = values.slice(0, state.originalN);
  const originalObjective = originalValues.reduce((sum, value, index) => sum + value * model.c[index], 0);
  const objectiveRow = state.tableau[state.basis.length];
  const basisSet = new Set(state.basis);
  const alternateColumns = [];

  for (let j = 0; j < state.originalN; j += 1) {
    if (!basisSet.has(j) && Math.abs(objectiveRow[j]) <= 1e-7) {
      alternateColumns.push(state.varNames[j]);
    }
  }

  return {
    status: phaseStatus,
    objective: cleanNumber(originalObjective),
    values: originalValues,
    allValues: values,
    varNames: state.varNames.slice(),
    varTypes: state.varTypes.slice(),
    basis: state.basis.slice(),
    alternateColumns,
  };
}

function solveTwoPhase(model) {
  const state = buildGeneralTableau(model);
  const iterations = [];
  let note = "";
  const hadArtificial = state.artificial.size > 0;

  if (hadArtificial) {
    const phaseOneCosts = Array(state.varNames.length).fill(0);
    state.artificial.forEach((index) => {
      phaseOneCosts[index] = -1;
    });
    setObjectiveRow(state, phaseOneCosts);
    const phaseOne = runPrimalSimplex(state, "Fase I");
    iterations.push(...phaseOne.iterations);

    const phaseOneValue = state.tableau[state.basis.length][state.varNames.length];
    if (phaseOne.status !== "optimal") {
      return {
        status: phaseOne.status,
        iterations,
        message: "La Fase I no pudo terminar en óptimo.",
        state,
      };
    }
    if (phaseOneValue < -1e-7) {
      return {
        status: "infeasible",
        iterations,
        message: "La suma de artificiales no pudo reducirse a cero; el modelo es infactible.",
        state,
      };
    }
    note = "Se ejecutó Fase I para construir una base factible y luego Fase II con la función objetivo original.";
    removeArtificialColumns(state);
  } else {
    note = "El modelo ya tenía una base factible con variables de holgura; se resolvió directamente con Simplex primal.";
  }

  const phaseTwoCosts = buildOriginalCosts(model, state.varNames.length, state.originalN);
  setObjectiveRow(state, phaseTwoCosts);
  const phaseTwo = runPrimalSimplex(state, hadArtificial ? "Fase II" : "Simplex Primal");
  iterations.push(...phaseTwo.iterations);

  if (phaseTwo.status === "unbounded") {
    return {
      status: "unbounded",
      iterations,
      message: `El modelo es no acotado por la variable ${state.varNames[phaseTwo.entering]}.`,
      state,
    };
  }
  if (phaseTwo.status !== "optimal") {
    return {
      status: phaseTwo.status,
      iterations,
      message: "Se alcanzó el límite de iteraciones antes de concluir.",
      state,
    };
  }

  return {
    ...analyzeFinalState(state, model, "optimal"),
    iterations,
    message: note,
    state,
  };
}

function canUsePrimal(model) {
  return model.constraints.every((constraint) => normalizeSign(constraint.sign) === "<=" && constraint.b >= -EPS);
}

function buildDualReadyTableau(model) {
  const originalN = model.c.length;
  const varNames = Array.from({ length: originalN }, (_, index) => `x${index + 1}`);
  const varTypes = Array.from({ length: originalN }, () => "decision");
  const basis = [];
  const rows = [];

  model.constraints.forEach((constraint, index) => {
    let sign = normalizeSign(constraint.sign);
    let a = constraint.a.slice();
    let b = constraint.b;
    if (sign === ">=") {
      a = a.map((value) => -value);
      b = -b;
      sign = "<=";
    }
    if (sign !== "<=") {
      throw new Error("El Simplex Dual directo no admite igualdades en el tablero inicial.");
    }
    const slackIndex = varNames.length;
    varNames.push(`s${index + 1}`);
    varTypes.push("slack");
    const row = Array(varNames.length + 1).fill(0);
    for (let j = 0; j < originalN; j += 1) row[j] = a[j];
    row[slackIndex] = 1;
    row[row.length - 1] = b;
    rows.forEach((existing) => existing.splice(existing.length - 1, 0, 0));
    rows.push(row);
    basis.push(slackIndex);
  });

  const state = {
    tableau: rows,
    basis,
    varNames,
    varTypes,
    artificial: new Set(),
    originalN,
  };
  const costs = buildOriginalCosts(model, varNames.length, originalN);
  setObjectiveRow(state, costs);
  return state;
}

function runDualSimplex(state) {
  const iterations = [];
  const cols = state.varNames.length;
  let count = 0;

  while (count < MAX_ITER) {
    let leaving = -1;
    let mostNegativeRhs = -EPS;
    for (let i = 0; i < state.basis.length; i += 1) {
      const rhs = state.tableau[i][cols];
      if (rhs < mostNegativeRhs) {
        mostNegativeRhs = rhs;
        leaving = i;
      }
    }

    if (leaving === -1) {
      iterations.push(
        snapshot(state, {
          phase: "Simplex Dual",
          title: "Factibilidad primal recuperada",
          summary: "Todos los lados derechos son no negativos y la fila Z mantiene factibilidad dual.",
        }),
      );
      return { status: "optimal", iterations };
    }

    let entering = -1;
    let bestRatio = Infinity;
    const ratios = Array(cols).fill(null);
    const objective = state.tableau[state.basis.length];

    for (let j = 0; j < cols; j += 1) {
      const coefficient = state.tableau[leaving][j];
      if (coefficient < -EPS) {
        const ratio = objective[j] / -coefficient;
        ratios[j] = ratio;
        if (ratio >= -EPS && ratio < bestRatio - EPS) {
          bestRatio = ratio;
          entering = j;
        }
      }
    }

    if (entering === -1) {
      iterations.push(
        snapshot(state, {
          phase: "Simplex Dual",
          title: "Infactibilidad detectada",
          leaving,
          ratios,
          summary: "Existe un lado derecho negativo sin columna candidata para recuperar factibilidad.",
        }),
      );
      return { status: "infeasible", iterations };
    }

    iterations.push(
      snapshot(state, {
        phase: "Simplex Dual",
        title: `Pivote dual ${count + 1}`,
        entering,
        leaving,
        pivotValue: state.tableau[leaving][entering],
        ratios,
        summary: `${state.varNames[state.basis[leaving]]} sale por tener el RHS más negativo; ${state.varNames[entering]} entra por el menor cociente dual.`,
      }),
    );
    pivot(state, leaving, entering);
    count += 1;
  }

  return { status: "iteration-limit", iterations };
}

function canUseDualInitial(state) {
  const objective = state.tableau[state.basis.length];
  for (let j = 0; j < state.varNames.length; j += 1) {
    if (objective[j] < -EPS) return false;
  }
  return true;
}

function solveDual(model) {
  try {
    const state = buildDualReadyTableau(model);
    if (!canUseDualInitial(state)) {
      const fallback = solveTwoPhase(model);
      fallback.message = "El tablero inicial no era dual-factible; se resolvió con Dos Fases para mantener validez.";
      return fallback;
    }
    const dual = runDualSimplex(state);
    if (dual.status !== "optimal") {
      return {
        status: dual.status,
        iterations: dual.iterations,
        message: dual.status === "infeasible" ? "El Simplex Dual detectó infactibilidad." : "Se alcanzó el límite de iteraciones.",
        state,
      };
    }
    return {
      ...analyzeFinalState(state, model, "optimal"),
      iterations: dual.iterations,
      message: "Se resolvió con Simplex Dual directo desde el tablero inicial.",
      state,
    };
  } catch (error) {
    const fallback = solveTwoPhase(model);
    fallback.message = `${error.message} Se resolvió con Dos Fases.`;
    return fallback;
  }
}

function solveModel(model, method = "two-phase") {
  const errors = validateModel(model);
  if (errors.length) return { status: "validation-error", errors, iterations: [] };

  if (method === "primal" && !canUsePrimal(model)) {
    const fallback = solveTwoPhase(model);
    fallback.message = "Simplex Primal requiere restricciones ≤ con RHS no negativo; se usó Dos Fases.";
    return fallback;
  }
  if (method === "dual") return solveDual(model);
  return solveTwoPhase(model);
}

function renderMathModel(model) {
  const objective = model.c
    .map((value, index) => `${formatNumber(value)}x${index + 1}`)
    .join(" + ")
    .replace(/\+ -/g, " - ");
  const constraints = model.constraints
    .map((constraint) => {
      const left = constraint.a
        .map((value, index) => `${formatNumber(value)}x${index + 1}`)
        .join(" + ")
        .replace(/\+ -/g, " - ");
      return `${left} ${constraint.sign} ${formatNumber(constraint.b)}`;
    })
    .join("<br>");
  return `${model.sense === "max" ? "Max" : "Min"} Z = ${objective}<br>${constraints}`;
}

function feasibleForGraph(point, model) {
  const [x, y] = point;
  if (x < -1e-7 || y < -1e-7) return false;
  return model.constraints.every((constraint) => {
    const value = constraint.a[0] * x + constraint.a[1] * y;
    const sign = normalizeSign(constraint.sign);
    if (sign === "<=") return value <= constraint.b + 1e-7;
    if (sign === ">=") return value >= constraint.b - 1e-7;
    return Math.abs(value - constraint.b) <= 1e-7;
  });
}

function uniquePoints(points) {
  const unique = [];
  points.forEach((point) => {
    if (!Number.isFinite(point[0]) || !Number.isFinite(point[1])) return;
    if (!unique.some((item) => Math.hypot(item[0] - point[0], item[1] - point[1]) < 1e-6)) {
      unique.push(point);
    }
  });
  return unique;
}

function lineIntersection(lineA, lineB) {
  const det = lineA.a * lineB.b - lineB.a * lineA.b;
  if (Math.abs(det) < EPS) return null;
  const x = (lineA.c * lineB.b - lineB.c * lineA.b) / det;
  const y = (lineA.a * lineB.c - lineB.a * lineA.c) / det;
  return [x, y];
}

function graphLines(model) {
  const lines = [
    { a: 1, b: 0, c: 0, label: "x₁ = 0", axis: true },
    { a: 0, b: 1, c: 0, label: "x₂ = 0", axis: true },
  ];
  model.constraints.forEach((constraint, index) => {
    lines.push({
      a: constraint.a[0],
      b: constraint.a[1],
      c: constraint.b,
      label: `R${index + 1}`,
      sign: normalizeSign(constraint.sign),
    });
  });
  return lines;
}

function feasibleVertices(model) {
  const lines = graphLines(model);
  const candidates = [];
  for (let i = 0; i < lines.length; i += 1) {
    for (let j = i + 1; j < lines.length; j += 1) {
      const point = lineIntersection(lines[i], lines[j]);
      if (point) candidates.push(point);
    }
  }
  return uniquePoints(candidates.filter((point) => feasibleForGraph(point, model)));
}

function lineSegmentInBox(line, maxX, maxY) {
  const points = [];
  const add = (x, y) => {
    if (x >= -1e-7 && x <= maxX + 1e-7 && y >= -1e-7 && y <= maxY + 1e-7) {
      points.push([Math.max(0, Math.min(maxX, x)), Math.max(0, Math.min(maxY, y))]);
    }
  };

  if (Math.abs(line.b) > EPS) {
    add(0, line.c / line.b);
    add(maxX, (line.c - line.a * maxX) / line.b);
  }
  if (Math.abs(line.a) > EPS) {
    add(line.c / line.a, 0);
    add((line.c - line.b * maxY) / line.a, maxY);
  }

  return uniquePoints(points).slice(0, 2);
}

function drawEmptyGraph(canvas, text) {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(640, rect.width * dpr);
  canvas.height = Math.max(360, rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = canvas.width / dpr;
  const height = canvas.height / dpr;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfaf6";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#68655d";
  ctx.font = "600 15px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(text, width / 2, height / 2);
}

function drawGraph(canvas, model, result) {
  if (!canvas) return { status: "empty", note: "No se encontró el lienzo del gráfico." };
  if (model.c.length !== 2) {
    drawEmptyGraph(canvas, "El gráfico está disponible solo para modelos con 2 variables.");
    return {
      status: "skip",
      note: "El modelo tiene más de 2 variables. El algoritmo funciona, pero la región ya no se puede mostrar en un plano 2D.",
    };
  }

  const vertices = feasibleVertices(model);
  const optimum = result.status === "optimal" && result.values?.length >= 2 ? result.values.slice(0, 2) : null;
  const boundsCandidates = vertices.concat(optimum ? [optimum] : []);
  model.constraints.forEach((constraint) => {
    const [a, b] = constraint.a;
    if (Math.abs(a) > EPS) boundsCandidates.push([constraint.b / a, 0]);
    if (Math.abs(b) > EPS) boundsCandidates.push([0, constraint.b / b]);
  });

  const positiveValues = boundsCandidates.flat().filter((value) => Number.isFinite(value) && value > 0);
  const maxBase = positiveValues.length ? Math.max(...positiveValues) : 10;
  const maxCoord = Math.max(5, maxBase * 1.25);
  const maxX = maxCoord;
  const maxY = maxCoord;

  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(640, rect.width * dpr);
  canvas.height = Math.max(380, rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = canvas.width / dpr;
  const height = canvas.height / dpr;
  const pad = { left: 56, right: 24, top: 24, bottom: 48 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const sx = (x) => pad.left + (x / maxX) * plotW;
  const sy = (y) => pad.top + plotH - (y / maxY) * plotH;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfaf6";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(21,21,21,0.08)";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#68655d";
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 5; i += 1) {
    const x = (maxX / 5) * i;
    const y = (maxY / 5) * i;
    ctx.beginPath();
    ctx.moveTo(sx(x), pad.top);
    ctx.lineTo(sx(x), pad.top + plotH);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad.left, sy(y));
    ctx.lineTo(pad.left + plotW, sy(y));
    ctx.stroke();
    ctx.fillText(formatNumber(x), sx(x), height - 22);
    ctx.textAlign = "right";
    ctx.fillText(formatNumber(y), pad.left - 10, sy(y));
    ctx.textAlign = "center";
  }

  ctx.strokeStyle = "rgba(21,21,21,0.72)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(pad.left, sy(0));
  ctx.lineTo(pad.left + plotW, sy(0));
  ctx.moveTo(sx(0), pad.top);
  ctx.lineTo(sx(0), pad.top + plotH);
  ctx.stroke();
  ctx.fillStyle = "#151515";
  ctx.font = "700 13px Inter, system-ui, sans-serif";
  ctx.fillText("x₁", pad.left + plotW - 8, height - 22);
  ctx.fillText("x₂", pad.left - 18, pad.top + 8);

  const polygon = vertices.slice();
  if (polygon.length >= 3 && result.status !== "unbounded") {
    const center = polygon.reduce((acc, point) => [acc[0] + point[0], acc[1] + point[1]], [0, 0]).map((value) => value / polygon.length);
    polygon.sort((p1, p2) => Math.atan2(p1[1] - center[1], p1[0] - center[0]) - Math.atan2(p2[1] - center[1], p2[0] - center[0]));
    ctx.beginPath();
    polygon.forEach((point, index) => {
      if (index === 0) ctx.moveTo(sx(point[0]), sy(point[1]));
      else ctx.lineTo(sx(point[0]), sy(point[1]));
    });
    ctx.closePath();
    ctx.fillStyle = "rgba(17,109,255,0.12)";
    ctx.fill();
    ctx.strokeStyle = "rgba(17,109,255,0.45)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  const colors = ["#116dff", "#137b53", "#b7791f", "#d94636", "#6b46c1", "#0f766e", "#c2410c"];
  graphLines(model)
    .filter((line) => !line.axis)
    .forEach((line, index) => {
      const segment = lineSegmentInBox(line, maxX, maxY);
      if (segment.length < 2) return;
      ctx.strokeStyle = colors[index % colors.length];
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx(segment[0][0]), sy(segment[0][1]));
      ctx.lineTo(sx(segment[1][0]), sy(segment[1][1]));
      ctx.stroke();
      ctx.fillStyle = colors[index % colors.length];
      ctx.font = "700 12px Inter, system-ui, sans-serif";
      ctx.fillText(`${line.label} ${line.sign}`, sx(segment[1][0]) - 18, sy(segment[1][1]) - 12);
    });

  vertices.forEach((point) => {
    ctx.beginPath();
    ctx.arc(sx(point[0]), sy(point[1]), 4, 0, Math.PI * 2);
    ctx.fillStyle = "#151515";
    ctx.fill();
  });

  if (optimum) {
    const objectiveLine = { a: model.c[0], b: model.c[1], c: result.objective };
    const segment = lineSegmentInBox(objectiveLine, maxX, maxY);
    if (segment.length === 2) {
      ctx.save();
      ctx.setLineDash([8, 6]);
      ctx.strokeStyle = "#d94636";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(sx(segment[0][0]), sy(segment[0][1]));
      ctx.lineTo(sx(segment[1][0]), sy(segment[1][1]));
      ctx.stroke();
      ctx.restore();
    }
    ctx.beginPath();
    ctx.arc(sx(optimum[0]), sy(optimum[1]), 7, 0, Math.PI * 2);
    ctx.fillStyle = "#d94636";
    ctx.fill();
    ctx.fillStyle = "#151515";
    ctx.font = "800 13px Inter, system-ui, sans-serif";
    ctx.fillText(`Óptimo (${formatNumber(optimum[0])}, ${formatNumber(optimum[1])})`, sx(optimum[0]) + 76, sy(optimum[1]) - 12);
  }

  if (result.status === "optimal") {
    return { status: "ok", note: "Gráfico generado: restricciones, región factible y punto óptimo." };
  }
  if (result.status === "unbounded") {
    return { status: "warning", note: "El gráfico muestra las restricciones, pero la región factible no queda cerrada: el modelo es no acotado." };
  }
  if (result.status === "infeasible") {
    return { status: "warning", note: "El gráfico muestra las rectas, pero no existe una región factible común." };
  }
  return { status: "ok", note: "Gráfico generado para el modelo de dos variables." };
}

function initApp() {
  const $ = (selector) => document.querySelector(selector);
  const objectiveSense = $("#objectiveSense");
  const methodSelect = $("#methodSelect");
  const variableCount = $("#variableCount");
  const constraintCount = $("#constraintCount");
  const objectiveFields = $("#objectiveFields");
  const constraintFields = $("#constraintFields");
  const modelForm = $("#modelForm");
  const messageBox = $("#messageBox");
  const statusTitle = $("#statusTitle");
  const summaryPanel = $("#summaryPanel");
  const iterationDetails = $("#iterationDetails");
  const iterationCounter = $("#iterationCounter");
  const prevIterationBtn = $("#prevIterationBtn");
  const nextIterationBtn = $("#nextIterationBtn");
  const graphCanvas = $("#graphCanvas");
  const graphStatus = $("#graphStatus");
  const graphNote = $("#graphNote");
  const exampleDialog = $("#exampleDialog");
  const exampleGrid = $("#exampleGrid");

  let renderedIterations = [];
  let currentIteration = 0;

  function clampCounts() {
    variableCount.value = Math.min(10, Math.max(1, Number(variableCount.value) || 1));
    constraintCount.value = Math.min(10, Math.max(1, Number(constraintCount.value) || 1));
  }

  function buildFields(existingModel = null) {
    clampCounts();
    const n = Number(variableCount.value);
    const m = Number(constraintCount.value);
    objectiveFields.innerHTML = "";
    objectiveFields.style.setProperty("--var-count", n);
    constraintFields.innerHTML = "";
    constraintFields.style.setProperty("--var-count", n);

    for (let j = 0; j < n; j += 1) {
      const label = document.createElement("label");
      label.innerHTML = `Coef. x${j + 1}<input type="number" step="any" data-role="objective" data-index="${j}" value="${existingModel?.c?.[j] ?? (j === 0 ? 3 : j === 1 ? 5 : 0)}">`;
      objectiveFields.appendChild(label);
    }

    for (let i = 0; i < m; i += 1) {
      const row = document.createElement("div");
      row.className = "constraint-row";
      row.style.setProperty("--var-count", n);
      const existing = existingModel?.constraints?.[i];
      for (let j = 0; j < n; j += 1) {
        const value = existing?.a?.[j] ?? 0;
        const label = document.createElement("label");
        label.innerHTML = `R${i + 1} · x${j + 1}<input type="number" step="any" data-role="constraint" data-row="${i}" data-col="${j}" value="${value}">`;
        row.appendChild(label);
      }
      const signLabel = document.createElement("label");
      signLabel.innerHTML = `Signo<select data-role="sign" data-row="${i}">
        <option value="<=" ${existing?.sign === "<=" ? "selected" : ""}>≤</option>
        <option value=">=" ${existing?.sign === ">=" ? "selected" : ""}>≥</option>
        <option value="=" ${existing?.sign === "=" ? "selected" : ""}>=</option>
      </select>`;
      row.appendChild(signLabel);
      const rhsLabel = document.createElement("label");
      rhsLabel.innerHTML = `RHS<input type="number" step="any" data-role="rhs" data-row="${i}" value="${existing?.b ?? 0}">`;
      row.appendChild(rhsLabel);
      constraintFields.appendChild(row);
    }
  }

  function readModel() {
    const n = Number(variableCount.value);
    const m = Number(constraintCount.value);
    const c = Array(n).fill(0);
    objectiveFields.querySelectorAll('[data-role="objective"]').forEach((input) => {
      c[Number(input.dataset.index)] = Number(input.value);
    });

    const constraints = Array.from({ length: m }, () => ({ a: Array(n).fill(0), sign: "<=", b: 0 }));
    constraintFields.querySelectorAll('[data-role="constraint"]').forEach((input) => {
      constraints[Number(input.dataset.row)].a[Number(input.dataset.col)] = Number(input.value);
    });
    constraintFields.querySelectorAll('[data-role="sign"]').forEach((select) => {
      constraints[Number(select.dataset.row)].sign = select.value;
    });
    constraintFields.querySelectorAll('[data-role="rhs"]').forEach((input) => {
      constraints[Number(input.dataset.row)].b = Number(input.value);
    });

    return {
      sense: objectiveSense.value,
      c,
      constraints,
    };
  }

  function setMessage(text, tone = "") {
    messageBox.className = `message-box ${tone}`.trim();
    messageBox.innerHTML = text;
  }

  function resetGraph() {
    drawEmptyGraph(graphCanvas, "Resuelve un problema con 2 variables para graficar.");
    graphStatus.textContent = "Disponible para 2 variables";
    graphNote.textContent =
      "Resuelve un problema con dos variables para visualizar restricciones, región factible y punto óptimo.";
  }

  function renderGraph(model, result) {
    const graphResult = drawGraph(graphCanvas, model, result);
    if (graphResult.status === "skip") graphStatus.textContent = "No disponible en 2D";
    else if (graphResult.status === "warning") graphStatus.textContent = "Revisar modelo";
    else graphStatus.textContent = "Gráfico actualizado";
    graphNote.textContent = graphResult.note;
  }

  function renderSummary(result, model) {
    summaryPanel.classList.remove("is-empty");
    if (result.status !== "optimal") {
      summaryPanel.innerHTML = `<div class="metric"><small>Estado</small><strong>${statusLabel(result.status)}</strong></div>`;
      return;
    }

    const basisSet = new Set(result.basis);
    const decisionHtml = result.values
      .map((value, index) => {
        const basic = basisSet.has(index) ? "Básica" : "No básica";
        return `<div class="metric"><small>x${index + 1} · ${basic}</small><strong>${formatNumber(value)}</strong></div>`;
      })
      .join("");

    const slackHtml = result.allValues
      .map((value, index) => ({ value, index, type: result.varTypes[index], name: result.varNames[index] }))
      .filter((item) => item.type === "slack" || item.type === "surplus")
      .map((item) => `<div class="metric"><small>${item.type === "slack" ? "Holgura" : "Exceso"} · ${item.name}</small><strong>${formatNumber(item.value)}</strong></div>`)
      .join("");

    const alternate = result.alternateColumns.length
      ? `<div class="metric"><small>Soluciones múltiples</small><strong>${result.alternateColumns.join(", ")}</strong></div>`
      : "";

    summaryPanel.innerHTML = `
      <div class="metric"><small>${model.sense === "max" ? "Valor máximo" : "Valor mínimo"} de Z</small><strong>${formatNumber(result.objective)}</strong></div>
      ${decisionHtml}
      ${slackHtml}
      ${alternate}
    `;
  }

  function statusLabel(status) {
    const labels = {
      optimal: "Solución óptima",
      infeasible: "Infactible",
      unbounded: "No acotado",
      "validation-error": "Datos inválidos",
      "iteration-limit": "Límite de iteraciones",
    };
    return labels[status] || status;
  }

  function renderIteration(index) {
    if (!renderedIterations.length) {
      iterationCounter.textContent = "Iteración 0 de 0";
      iterationDetails.innerHTML = "<p>Los tableros Simplex se mostrarán después de resolver.</p>";
      prevIterationBtn.disabled = true;
      nextIterationBtn.disabled = true;
      return;
    }

    currentIteration = Math.min(Math.max(index, 0), renderedIterations.length - 1);
    const item = renderedIterations[currentIteration];
    iterationCounter.textContent = `Iteración ${currentIteration + 1} de ${renderedIterations.length}`;
    prevIterationBtn.disabled = currentIteration === 0;
    nextIterationBtn.disabled = currentIteration === renderedIterations.length - 1;

    const cols = item.varNames.length;
    const ratioList = item.ratios?.length
      ? `<p><strong>Cocientes:</strong> ${item.ratios.map((ratio, row) => `F${row + 1}: ${ratio === null ? "—" : formatNumber(ratio)}`).join(" · ")}</p>`
      : "";

    const header = ["Base", ...item.varNames, "RHS"];
    const bodyRows = item.tableau.map((row, rowIndex) => {
      const isObjective = rowIndex === item.basis.length;
      const baseName = isObjective ? "Z" : item.varNames[item.basis[rowIndex]];
      const cells = row.map((value, colIndex) => {
        const isPivotCol = colIndex === item.entering;
        const isPivotRow = rowIndex === item.leaving;
        const isPivotCell = isPivotCol && isPivotRow;
        const classes = [isPivotCol ? "pivot-col" : "", isPivotCell ? "pivot-cell" : ""].filter(Boolean).join(" ");
        return `<td class="${classes}">${formatNumber(value)}</td>`;
      });
      return `<tr class="${rowIndex === item.leaving ? "pivot-row" : ""}"><td>${baseName}</td>${cells.join("")}</tr>`;
    });

    iterationDetails.innerHTML = `
      <div class="iteration-summary">
        <p class="eyebrow">${item.phase}</p>
        <h3>${item.title}</h3>
        <p>${item.summary}</p>
        ${
          item.entering !== null
            ? `<p><strong>Variable entrante:</strong> ${item.varNames[item.entering]}${
                item.leaving !== null ? ` · <strong>Variable saliente:</strong> ${item.varNames[item.basis[item.leaving]]}` : ""
              }</p>`
            : ""
        }
        ${item.pivotValue !== null ? `<p><strong>Elemento pivote:</strong> ${formatNumber(item.pivotValue)}</p>` : ""}
        ${ratioList}
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr>${header.map((cell) => `<th>${cell}</th>`).join("")}</tr></thead>
          <tbody>${bodyRows.join("")}</tbody>
        </table>
      </div>
    `;
  }

  function solveFromForm() {
    const model = readModel();
    const result = solveModel(model, methodSelect.value);
    renderedIterations = result.iterations || [];
    statusTitle.textContent = statusLabel(result.status);

    if (result.status === "validation-error") {
      setMessage(result.errors.join("<br>"), "danger");
      renderSummary(result, model);
      resetGraph();
    } else if (result.status === "optimal") {
      const alternate = result.alternateColumns?.length
        ? " Además, hay indicio de soluciones óptimas múltiples."
        : "";
      setMessage(`${result.message || "Modelo resuelto correctamente."}${alternate}<br><br><strong>Modelo:</strong><br>${renderMathModel(model)}`, "success");
      renderSummary(result, model);
      renderGraph(model, result);
    } else if (result.status === "infeasible") {
      setMessage(result.message || "El problema es infactible.", "danger");
      renderSummary(result, model);
      renderGraph(model, result);
    } else if (result.status === "unbounded") {
      setMessage(result.message || "El problema es no acotado.", "warning");
      renderSummary(result, model);
      renderGraph(model, result);
    } else {
      setMessage(result.message || "No se pudo concluir el proceso.", "warning");
      renderSummary(result, model);
      renderGraph(model, result);
    }
    renderIteration(0);
  }

  function loadExample(example) {
    objectiveSense.value = example.model.sense;
    methodSelect.value = example.model.method;
    variableCount.value = example.model.c.length;
    constraintCount.value = example.model.constraints.length;
    buildFields(example.model);
    if (exampleDialog.open) exampleDialog.close();
    setMessage(`Ejemplo cargado: <strong>${example.title}</strong>. Presiona Resolver modelo.`, "");
  }

  function renderExamples() {
    exampleGrid.innerHTML = examples
      .map(
        (example) => `
          <button class="example-card" type="button" data-example="${example.id}">
            <small>${example.title}</small>
            <strong>${example.model.sense === "max" ? "Maximizar" : "Minimizar"}</strong>
            <span>${example.description}</span>
          </button>
        `,
      )
      .join("");
    exampleGrid.querySelectorAll("[data-example]").forEach((button) => {
      button.addEventListener("click", () => {
        const example = examples.find((item) => item.id === button.dataset.example);
        loadExample(example);
      });
    });
  }

  variableCount.addEventListener("change", () => buildFields(readModel()));
  constraintCount.addEventListener("change", () => buildFields(readModel()));
  $("#addVariableBtn").addEventListener("click", () => {
    variableCount.value = Math.min(10, Number(variableCount.value) + 1);
    buildFields(readModel());
  });
  $("#addConstraintBtn").addEventListener("click", () => {
    constraintCount.value = Math.min(10, Number(constraintCount.value) + 1);
    buildFields(readModel());
  });
  $("#resetBtn").addEventListener("click", () => {
    variableCount.value = 2;
    constraintCount.value = 3;
    objectiveSense.value = "max";
    methodSelect.value = "two-phase";
    buildFields();
    renderedIterations = [];
    summaryPanel.className = "summary-panel is-empty";
    summaryPanel.innerHTML = "<p>La solución óptima aparecerá aquí.</p>";
    statusTitle.textContent = "Listo para resolver";
    setMessage("Carga un ejemplo o escribe tu propio problema. Los campos aceptan enteros, decimales y negativos.");
    resetGraph();
    renderIteration(0);
  });
  modelForm.addEventListener("submit", (event) => {
    event.preventDefault();
    solveFromForm();
  });
  $("#solveTopBtn").addEventListener("click", solveFromForm);
  $("#loadExampleBtn").addEventListener("click", () => exampleDialog.showModal());
  prevIterationBtn.addEventListener("click", () => renderIteration(currentIteration - 1));
  nextIterationBtn.addEventListener("click", () => renderIteration(currentIteration + 1));

  buildFields();
  renderExamples();
  resetGraph();
}

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", initApp);
}

if (typeof module !== "undefined") {
  module.exports = {
    solveModel,
    examples,
    formatNumber,
  };
}
