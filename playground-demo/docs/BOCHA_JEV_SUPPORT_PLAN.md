# Jev Bocha comparison support plan

## Outcome

Add Jev Bocha as a second challenger in the local playground. Each run compares exactly two providers:

- Local MLX **or** Jev Bocha
- Jev official as the fixed baseline

Three-way comparison is intentionally deferred so the current table remains readable.

## Integration contract

- Reuse `typesafe-sdk==0.7.1`.
- Send Bocha evaluations to `https://jev.bocha.cn/v1/systemone`.
- Default to model `bocha-jev-v1`.
- Keep the Bocha API key in the loopback Python process only. Never persist it in browser storage or return it from health/config APIs.
- Accept `BOCHA_JEV_API_KEY` and `BOCHA_JEV_MODEL` as optional startup configuration.
- Allow runtime setup through a password field; clear the field immediately after a successful save.
- Restrict the configurable base URL to the documented HTTPS Bocha endpoint to prevent server-side request forgery.

## Implementation

1. Add sanitized Bocha health, in-memory configuration, and evaluation methods to `server.py`.
2. Add `POST /api/config/bocha` and `POST /api/evaluate/bocha`.
3. Replace the two independent target checkboxes with a Local MLX / Jev Bocha challenger selector. Jev official remains mandatory.
4. Make table, rendered response, raw JSON, comparison labels, and timing summary follow the selected challenger.
5. Add a Jev Bocha section to Setup for API key, fixed endpoint, and model.
6. Document setup and security boundaries in the README.

## Validation

- Unit-test SDK configuration, response timing, credential separation, secret redaction, and URL validation.
- Test static UI wiring and ensure the API key is never written to `localStorage`.
- Exercise the setup and challenger-selection journey in a browser against an isolated local server.
- Run the playground tests and the repository-required validation commands.
