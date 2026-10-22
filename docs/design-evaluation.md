# EvoKit Design Evaluation

An independent assessment of EvoKit's architecture and design decisions.

## Executive Summary

EvoKit is a well-architected evolutionary computation library optimized for a specific use case: **agent-driven variation in evolutionary search**. The design is minimal, focused, and makes strong choices that favor clarity over generality. The central bet—that intelligent agents as variation operators outperform random mutation—shapes every architectural decision.

## Core Design Strengths

### 1. Single Responsibility Per Component

Each of the five base classes owns exactly one decision:

| Component | Controls | Explicitly Does Not |
|-----------|----------|---------------------|
| Evaluator | Fitness scoring, validity | Storage, ranking, survival |
| Archive | Population membership, ranking | History, parent selection |
| Strategy | Parent selection | Genome creation, fitness |
| Operator | Genome variation, cost tracking | Real fitness, survival |
| Store | Persistence, locking | Any evolutionary logic |

This separation is enforced consistently. The archive never peeks at the operator's draft evaluations. The strategy reads only `niches()` and `members()`, so it works with any archive topology. The operator returns individuals *without* fitness, maintaining the evaluator's monopoly on scoring.

### 2. Validity Belongs to the Evaluator

The decision to make `status="failed"` an evaluator responsibility rather than a gate is subtle but correct. It means:

- Archives only rank; they never reject on validity grounds
- The evaluator is the single source of truth for "is this genome acceptable?"
- Failed genomes are still recorded (useful for debugging, analysis)

This avoids the common pitfall where validity checks proliferate across components.

### 3. Async-First, Sync-Friendly

Operators and strategies are async methods, but `Experiment` and `Engine` expose a sync API via a background event loop. This threading model:

- Supports long-running agent sessions without blocking
- Works in Jupyter, CLI, MCP handlers without async boilerplate
- Keeps the core loop readable: `selected → create → evaluate → admit → tell`

### 4. Cost as a First-Class Concern

Every operation tracks cost in `info["usage"]` with a consistent schema (`cost_usd`, `tokens`, `draft_evals`, `seconds`). The budget is:

- Charged when individuals are evaluated (not when created)
- Counts draft evaluations within operators
- Caps both final evals and dollar spend

This is essential for the agent-vs-LLM comparison the library exists to make.

### 5. The `tell()` Feedback Loop

Both Strategy and Operator receive admission feedback via `tell([individual], [admitted])`. This enables:

- `Choice` operator adapting weights by admission rate
- Strategies learning from rejection patterns
- Supervisors redirecting after stagnation

Without this hook, components would be stateless and unable to adapt.

### 6. Genome Type Flexibility

The `to_json(put_blob)` / `from_json(data, get_blob)` protocol for custom genome types is elegant:

- JSON primitives work as-is
- Complex types (Files, neural networks) serialize to `{"$type": "module:Class", "data": ...}`
- Blobs are content-addressed, so identical file contents share storage

### 7. Multi-Surface Interface

A single `@command` decorator generates CLI, MCP tools, Python callables, and SKILL.md. This is unusual and useful: the same experiment can be driven by a human at a terminal, an agent via MCP, or a script via `evokit.tools()`.

## Design Tensions Worth Noting

### 1. Minimal Archive Implementations

Only `BestArchive` and `PopulationArchive` exist. The design doc mentions `GridArchive` and islands as future work. For the stated use case (code optimization with agents), this is fine—most runs use hill-climbing (`BestArchive(ties="accept")`).

However, multi-objective optimization (Pareto fronts), diversity maintenance (novelty search), and constraint handling would need new archive types.

### 2. Strategy Simplicity

Only `Greedy` exists. This is sufficient when the operator is an intelligent agent (the agent brings its own exploration), but limits use with random operators where selection pressure matters.

### 3. No Built-In Parallelism

`Engine.batch_size` creates multiple individuals concurrently within one process. True distributed execution (multiple machines, task queues) requires external orchestration. The `DirectoryStore` with file locking enables multi-process parallelism, but it's manual.

### 4. Evaluation Caching Absent

Identical genomes are re-evaluated. For expensive evaluations, a content-addressed cache in the Store would help. The genome serialization machinery (`to_json`) could support this with minimal changes.

### 5. Operators Assume Agent/LLM Context

The existing operators (`LLMOperator`, `CodingAgentOperator`, `DiffOperator`) are LLM-centric. Standard mutation/crossover for numeric or discrete genomes would need new implementations.

This isn't a flaw—it reflects the library's purpose—but users expecting traditional EA operators will need to write them.

## The Files Subsystem

The `evokit.files` module deserves separate mention. It solves a real problem: evolving code in a project directory while keeping the evaluator immune to tampering.

- `Project` splits files into genome (evolvable) and environment (protected)
- `ScriptEvaluator` writes to a fresh directory and runs a command
- `Files` is immutable with content-addressed blob storage

The pattern of "write genome to directory → run evaluation script → read back" is clean and sandboxed. The `$EVOKIT_DRAFT_EVAL` mechanism for agents to test drafts is practical.

## Comparative Position

EvoKit is not trying to be DEAP, Optuna, or a general-purpose EA framework. It's a focused library for a specific hypothesis: agent operators > random operators at equal cost.

This focus leads to:

- **Strong:** Minimal API, clear cost tracking, async operators, multi-surface interface
- **Absent:** Visualization, algorithm zoo, hyperparameter tuning, Pareto optimization

The design is appropriate for its stated goal.

## Recommendations

1. **Evaluation caching**: Add optional content-addressed caching in Store for deterministic evaluators.

2. **Convergence detection**: The `strategy.done()` hook exists but isn't utilized. Consider plateau detection based on best fitness stagnation.

3. **Operator composition**: Beyond `Choice`, consider `Pipeline` (sequence of operators) or `Fallback` (try A, on failure try B).

4. **Archive events**: The event log tracks `evaluate` and `add`. Consider also logging archive state transitions (new best, population turnover).

5. **Batch evaluation**: For expensive evaluators with batching support (GPU), consider a `batch_evaluate(genomes)` method.

## Conclusion

EvoKit makes the right tradeoffs for its purpose. The architecture is clean, the responsibilities are clear, and the implementation is honest about what it doesn't do. The `tell()` feedback loop and cost-tracking infrastructure are the key innovations that enable the agent-vs-LLM experiments the library exists to run.

The design would be inappropriate for a general EA library, but EvoKit isn't trying to be one. For "can a coding agent beat one-shot LLM mutation?"—it's well-suited to answer that question.
