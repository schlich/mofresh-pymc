import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This notebook demonstrates how to get a PyMC progress bar to work in Marimo using an example from the PyMC docs.

    The main trick is to update the AnyWidget using the callback function passed to `pm.sample()`. The `PyMCProgress` class handles the logic of updating the progress bar and displaying it using `mofresh.HTMLRefreshWidget`.

    Implementations of the `trace` and `draw` objects that are passed to the callback function automatically by PyMC can be found here:

    - [trace](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.backends.NDArray.html)
    - [draw](https://github.com/pymc-devs/pymc/blob/360cb6edde9ccba306c0e046d9576c936fa4e571/pymc/sampling/parallel.py#L414)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    tune = mo.ui.number(value=500, label="Tuning draws per chain")
    draws = mo.ui.number(value=500, label="Sample draws per chain")
    chains = mo.ui.number(4, label="Number of chains")
    mo.hstack([tune, draws, chains])
    return chains, draws, tune


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""The `PyMCProgress` class is LLM slop but works pretty well!  could definitely be refactored some..."""
    )
    return


@app.cell(hide_code=True)
def _(HTMLRefreshWidget, math, sys):
    class PyMCProgress:
        """
        Displays PyMC progress using mofresh's HTMLRefreshWidget.
        Updates an HTML table with per-chain and overall progress.
        """

        def __init__(
            self,
            num_chains: int,
            tune_steps_per_chain: int,
            draw_steps_per_chain: int,
        ):
            """
            Initializes the mofresh-based progress tracker.

            Args:
                num_chains (int): The total number of chains being run.
                tune_steps_per_chain (int): Number of tuning steps for each chain.
                draw_steps_per_chain (int): Number of drawing (sampling) steps for each chain.
            """
            self.num_chains = num_chains
            self.tune_steps_per_chain = tune_steps_per_chain
            self.draw_steps_per_chain = draw_steps_per_chain

            self.total_overall_iterations = (
                tune_steps_per_chain + draw_steps_per_chain
            ) * num_chains
            self.total_callbacks_received = 0

            self.chains_status = []
            for i in range(num_chains):
                self.chains_status.append(
                    {
                        "id": i,
                        "stage": "Tuning",
                        "current_steps_in_stage": 0,
                        "total_steps_in_stage": self.tune_steps_per_chain,
                        "completed_tuning": False,
                        "completed_sampling": False,
                        "divergences_sampling": 0,
                        "leapfrog_steps": 0,
                    }
                )

            # Create the widget that will display the HTML
            self.display_widget = HTMLRefreshWidget()
            self._initial_html_set = True

        def _generate_html_progress(self, is_final: bool = False) -> str:
            """
            Generates an HTML string representing the current progress.
            """
            html_parts = [
                '<table border="1" style="width:100%; border-collapse: collapse; font-family: monospace;">'
            ]
            html_parts.append(
                "<tr><th>Chain</th><th>Stage</th><th>Progress</th><th>%</th><th>Div(S)</th><th>LF Steps</th></tr>"
            )

            total_divergences_all_chains_sampling = 0

            for chain_stat in self.chains_status:
                stage = chain_stat["stage"]
                current = chain_stat["current_steps_in_stage"]
                total_stage = chain_stat["total_steps_in_stage"]
                divergences_sampling = chain_stat["divergences_sampling"]
                leapfrogs = chain_stat["leapfrog_steps"]

                total_divergences_all_chains_sampling += divergences_sampling

                if (
                    stage == "Done"
                ):  # Ensure 'Done' shows full progress for that stage
                    if (
                        chain_stat["completed_tuning"]
                        and not chain_stat["completed_sampling"]
                    ):
                        current = self.draw_steps_per_chain
                        total_stage = self.draw_steps_per_chain
                    elif not chain_stat["completed_tuning"]:
                        current = self.tune_steps_per_chain
                        total_stage = self.tune_steps_per_chain
                    else:
                        current = total_stage

                progress_frac = 0.0
                if total_stage > 0:
                    progress_frac = min(1.0, max(0.0, current / total_stage))
                elif current > 0:
                    progress_frac = 1.0

                percent_complete_stage = progress_frac * 100

                # Simple bar (optional, can be removed or enhanced with CSS)
                bar_width_chars = 20  # Number of characters for the text bar
                filled_chars = math.floor(bar_width_chars * progress_frac)
                bar_color = "grey"  # Default or for 'Done' if not fully sampled
                if stage == "Tuning":
                    bar_color = "gold"
                elif stage == "Sampling":
                    bar_color = "mediumseagreen"
                elif (
                    stage == "Done" and current >= total_stage
                ):  # Ensure 'Done' is fully green if completed
                    bar_color = "mediumseagreen"

                filled_bar_html = (
                    f'<span style="color:{bar_color};">{"█" * filled_chars}</span>'
                )
                empty_bar_html = f'<span style="color:#e0e0e0;">{"─" * (bar_width_chars - filled_chars)}</span>'
                bar_str_html = filled_bar_html + empty_bar_html

                html_parts.append(
                    f"<tr>"
                    f"<td style='text-align:center;'>C{chain_stat['id'] + 1}</td>"
                    f"<td style='text-align:center;'>{stage}</td>"
                    f"<td style='text-align:left; padding-left: 5px;'>{bar_str_html} {current}/{total_stage}</td>"
                    f"<td style='text-align:right; padding-right: 5px;'>{percent_complete_stage:.1f}%</td>"
                    f"<td style='text-align:center;'>{divergences_sampling}</td>"
                    f"<td style='text-align:center;'>{leapfrogs}</td>"
                    f"</tr>"
                )

            html_parts.append("</table>")

            # Overall Progress
            if self.total_overall_iterations > 0:
                overall_frac = 0.0
                if self.total_overall_iterations > 0:
                    overall_frac = min(
                        1.0,
                        self.total_callbacks_received
                        / self.total_overall_iterations,
                    )
                overall_percent = overall_frac * 100

                overall_summary_text = f"Overall Progress: {self.total_callbacks_received}/{self.total_overall_iterations} ({overall_percent:.1f}%)"
                if is_final:
                    if (
                        self.total_callbacks_received
                        >= self.total_overall_iterations
                    ):
                        overall_summary_text = f"Overall: Complete ({overall_percent:.1f}%). Total Sampling Divergences: {total_divergences_all_chains_sampling}"
                    else:
                        overall_summary_text = f"Overall: {self.total_callbacks_received}/{self.total_overall_iterations} ({overall_percent:.1f}%) Ended. Total Sampling Divergences: {total_divergences_all_chains_sampling}"

                html_parts.append(
                    f"<p style='font-family: monospace; margin-top: 5px;'>{overall_summary_text}</p>"
                )

            return "".join(html_parts)

        def _update_display(self, is_final: bool = False):
            """Generates HTML and updates the widget."""
            current_html = self._generate_html_progress(is_final=is_final)
            self.display_widget.html = current_html
            self._initial_html_set = True

        def callback(self, trace, draw) -> None:
            """
            Callback function to be invoked by PyMC at each step.
            """
            self.total_callbacks_received += 1
            chain_idx = draw.chain

            if not (0 <= chain_idx < self.num_chains):
                # Handle error, perhaps log it. For now, skip update.
                print(
                    f"Warning: Invalid chain index {chain_idx} received in MofreshPymcProgress.",
                    file=sys.stderr,
                )
                return

            chain_stat = self.chains_status[chain_idx]

            is_tuning_sample = False
            diverged_this_step = False
            leapfrog_steps_this_step = 0

            stats_to_check = (
                draw.stats[0]
                if isinstance(draw.stats, list) and draw.stats
                else draw.stats
            )

            if isinstance(stats_to_check, dict):
                is_tuning_sample = stats_to_check.get("tune", False)
                if stats_to_check.get("diverging", False) or stats_to_check.get(
                    "divergence", False
                ):
                    diverged_this_step = True
                leapfrog_steps_this_step = stats_to_check.get("n_steps", 0)

            if diverged_this_step and not is_tuning_sample:
                chain_stat["divergences_sampling"] += 1
            chain_stat["leapfrog_steps"] += leapfrog_steps_this_step

            if not chain_stat["completed_tuning"]:
                if (
                    not is_tuning_sample
                    and chain_stat["current_steps_in_stage"]
                    < self.tune_steps_per_chain
                ):
                    chain_stat["current_steps_in_stage"] = (
                        self.tune_steps_per_chain
                    )

                if is_tuning_sample:
                    chain_stat["stage"] = "Tuning"
                    chain_stat["total_steps_in_stage"] = self.tune_steps_per_chain
                    chain_stat["current_steps_in_stage"] += 1

                if (
                    chain_stat["current_steps_in_stage"]
                    >= self.tune_steps_per_chain
                ):
                    chain_stat["completed_tuning"] = True
                    chain_stat["stage"] = "Sampling"
                    chain_stat["current_steps_in_stage"] = 0
                    chain_stat["total_steps_in_stage"] = self.draw_steps_per_chain
                    if is_tuning_sample:
                        self._update_display()
                        return

            if (
                chain_stat["completed_tuning"]
                and not chain_stat["completed_sampling"]
            ):
                if is_tuning_sample:
                    chain_stat["stage"] = "Sampling"
                    chain_stat["total_steps_in_stage"] = self.draw_steps_per_chain
                else:
                    chain_stat["stage"] = "Sampling"
                    chain_stat["total_steps_in_stage"] = self.draw_steps_per_chain
                    chain_stat["current_steps_in_stage"] += 1

                if (
                    chain_stat["current_steps_in_stage"]
                    >= self.draw_steps_per_chain
                ):
                    chain_stat["completed_sampling"] = True
                    chain_stat["stage"] = "Done"

            elif chain_stat["completed_sampling"]:
                chain_stat["stage"] = "Done"
                chain_stat["current_steps_in_stage"] = self.draw_steps_per_chain
                chain_stat["total_steps_in_stage"] = self.draw_steps_per_chain

            self._update_display()

        def finalize(self) -> None:
            """
            Updates the display with the final progress state.
            """
            if not self._initial_html_set and self.total_overall_iterations == 0:
                self.display_widget.html = "<p style='font-family: monospace;'>No iterations performed.</p>"
                return
            if not self._initial_html_set and self.total_overall_iterations > 0:
                # If finalize is called before any callback (e.g. error before sampling starts)
                self._update_display(is_final=False)  # Show initial 0% state

            for chain_stat in self.chains_status:
                if chain_stat["completed_sampling"]:
                    chain_stat["stage"] = "Done"
                elif chain_stat["completed_tuning"]:
                    chain_stat["stage"] = (
                        "Sampling"  # Could be 'Sampling (Ended)' if you prefer
                    )
                else:
                    chain_stat["stage"] = "Tuning"  # Could be 'Tuning (Ended)'

            self._update_display(is_final=True)

        def reset(self):
            """Resets the internal state of the progress tracker for a new run."""
            self.total_overall_iterations = (
                self.tune_steps_per_chain + self.draw_steps_per_chain
            ) * self.num_chains
            self.total_callbacks_received = 0
            self.chains_status = []
            for i in range(self.num_chains):
                self.chains_status.append(
                    {
                        "id": i,
                        "stage": "Tuning",
                        "current_steps_in_stage": 0,
                        "total_steps_in_stage": self.tune_steps_per_chain,
                        "completed_tuning": False,
                        "completed_sampling": False,
                        "divergences_sampling": 0,
                        "leapfrog_steps": 0,
                    }
                )
            self._initial_html_set = (
                False  # Ensure the display updates from scratch
            )

        def __enter__(self):
            """
            Enter the runtime context. Sets initial HTML.
            """
            self.reset()  # Reset state at the beginning of each 'with' block
            self._update_display()  # Update display to show initial (reset) state
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Exit the runtime context. Ensures finalize() is called.
            """
            self.finalize()
            return False  # Re-raise any exceptions
    return (PyMCProgress,)


@app.cell
def _(PyMCProgress, chains, draws, tune):
    progress_tracker = PyMCProgress(
        num_chains=chains.value,
        tune_steps_per_chain=tune.value,
        draw_steps_per_chain=draws.value,
    )
    return (progress_tracker,)


@app.cell
def _(progress_tracker):
    progress_tracker.display_widget
    return


@app.cell
def _(
    chains,
    draws,
    generative_model,
    pm,
    progress_tracker,
    synthetic_y,
    tune,
):
    with progress_tracker:
        with pm.observe(
            generative_model, {"plant growth": synthetic_y}
        ) as inference_model:
            idata = pm.sample(
                tune=tune.value,
                draws=draws.value,
                chains=chains.value,
                callback=progress_tracker.callback,  # <-- this is where you put the callback
            )
    return


@app.cell(hide_code=True)
def _(pm):
    # Taking draws from a normal distribution
    x_dist = pm.Normal.dist(shape=(100, 3))
    x_data = pm.draw(x_dist)

    # Define coordinate values for all dimensions of the data
    coords = {
        "trial": range(100),
        "features": ["sunlight hours", "water amount", "soil nitrogen"],
    }

    # Define generative model
    with pm.Model(coords=coords) as generative_model:
        x = pm.Data("x", x_data, dims=["trial", "features"])

        # Model parameters
        betas = pm.Normal("betas", dims="features")
        sigma = pm.HalfNormal("sigma")

        # Linear model
        mu = x @ betas

        # Likelihood
        # Assuming we measure deviation of each plant from baseline
        plant_growth = pm.Normal("plant growth", mu, sigma, dims="trial")


    # Generating data from model by fixing parameters
    fixed_parameters = {
        "betas": [5, 20, 2],
        "sigma": 0.5,
    }
    with pm.do(generative_model, fixed_parameters) as synthetic_model:
        synthetic_y = (
            pm.sample_prior_predictive().prior["plant growth"].sel(draw=0, chain=0)
        )
    return generative_model, synthetic_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Trying a more robust solution with D3 plots, I can't seem to get the plot to update live. However, rendering the widget after the MCMC run plots the traces as expected. Seems like something is going wrong somewhere in the area of `el.appendChild(chart)`; maybe the HTML isn't being injected right."""
    )
    return


@app.cell
def _(Draw, NDArray, anywidget, traitlets):
    class D3Widget(anywidget.AnyWidget):
        _esm = """
        import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm";

        function render({ model, el }) {
          let data = () => model.get("data");
          let chart = Plot.plot({
                marks: [
                  Plot.lineY(data(), {x: "Sample", y: "Beta1", stroke: "Chain"})
                ]
              });

            model.on("change:data", () => {
              chart = Plot.plot({
                marks: [
                    Plot.lineY(data(), {x: "Sample", y: "Beta1", stroke: "Chain"})
                  ]
              });
            });
            el.appendChild(chart);
        }

        export default { render };
        """

        data = traitlets.List().tag(sync=True)


    class PyMCWidget:
        def __init__(self):
            self.widget = D3Widget()

        def callback(self, trace: NDArray, draw: Draw) -> None:
            betas = draw.point["betas"]
            self.widget.data.append(
                {
                    "Beta1": betas[0],
                    "Sample": draw.draw_idx,
                    "Chain": draw.chain,
                }
            )
    return (PyMCWidget,)


@app.cell
def _(PyMCWidget, mo):
    # This doesnt update live....
    trace_plotting = PyMCWidget()
    mo.ui.anywidget(trace_plotting.widget)
    return (trace_plotting,)


@app.cell
def _(chains, draws, generative_model, pm, synthetic_y, trace_plotting, tune):
    with pm.observe(generative_model, {"plant growth": synthetic_y}) as model:
        output = pm.sample(
            tune=tune.value,
            draws=draws.value,
            chains=chains.value,
            callback=trace_plotting.callback,
        )
    return (output,)


@app.cell
def _(mo, output, trace_plotting):
    # ... but plotting after sampling works as expected

    output # placeholder for cell run order
    mo.ui.anywidget(trace_plotting.widget)
    return


@app.cell
def _(trace_plotting):
    trace_plotting.widget.data
    return


@app.cell
def _():
    import marimo as mo
    from mofresh import HTMLRefreshWidget
    import pymc as pm
    import math
    import anywidget
    import traitlets
    from pymc.backends import NDArray
    from pymc.sampling.parallel import Draw
    from time import sleep
    return Draw, HTMLRefreshWidget, NDArray, anywidget, math, mo, pm, traitlets


if __name__ == "__main__":
    app.run()
