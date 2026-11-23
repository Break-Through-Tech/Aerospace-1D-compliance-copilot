
import { useState } from "react";
import "./App.css";

function App() {
  const [requirement, setRequirement] = useState("");
  const [mode, setMode] = useState("baseline");

  function handleRunAnalysis() {
    if (!requirement.trim()) {
      return;
    }
  
    alert(
      `Analysis started!\n\nMode: ${mode}\n\nRequirement:\n${requirement}`
    );
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h2>Compliance Copilot</h2>

        <nav>
          <a className="active" href="#">
            Dashboard
          </a>
          <a href="#">
            Results
          </a>
          <a href="#">
            Evaluation
          </a>
        </nav>
      </aside>

      <main className="main-content">
        <header>
          <h1>Requirements Analysis</h1>
          <p>
            Analyze software requirements against
            NASA NPR 7150.2D.
          </p>
        </header>

        <section className="input-section">
          <h2>Enter Requirements</h2>

          <textarea
            placeholder="Paste a software requirement here..."
            value={requirement}
            onChange={(event) =>
              setRequirement(event.target.value)
            }
          />

          <div className="controls">
            <label htmlFor="mode">Analysis Mode</label>

            <select
              id="mode"
              value={mode}
              onChange={(event) =>
                setMode(event.target.value)
              }
            >
              <option value="baseline">Baseline</option>
              <option value="agent">Agent</option>
            </select>

            <button
              onClick={handleRunAnalysis}
              disabled={!requirement.trim()}
            >
              Run Analysis
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
