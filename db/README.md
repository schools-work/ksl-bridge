# db/

| File | Purpose |
|---|---|
| `ksl_bridge_v1.sql` | **Canonical schema**: 15 schemas, RLS, functions, views, grants, seed reference data |
| `seeds/dev_users.sql` | Local development users and the `ksl_app_login` role (never run in production) |
| `seeds/demo_restaurant.sql` | Demo venue, menu knowledge base, venue vocabulary, enabled tools |
| `checks/schema_invariants.sql` | Executable tests for the invariants. Runs inside a transaction and **rolls back** |

## Load
```bash
make schema          # psql -f db/ksl_bridge_v1.sql && psql -f db/checks/schema_invariants.sql
make seed
```

## Roles
`ksl_migrator` (owner/DDL) · `ksl_app` (gateway) · `ksl_worker` (inference) · `ksl_trainer` (training/eval) · `ksl_readonly` (dashboards). Log-in roles are created per environment and granted one of these.

## Per-request GUCs (RLS is fail-closed)
```sql
SET LOCAL app.current_user_id  = '<uuid>';
SET LOCAL app.current_roles    = 'signer,annotator';
SET LOCAL app.current_venue_id = '<uuid>';      -- operators and devices
```
`app/platform/db.py` does this for every request transaction.

## Partition maintenance
Run monthly (the scheduler does this): `SELECT ops.create_monthly_partitions('conversation.utterances', now()::date, 3);` and the same for `inference.requests` and `audit.events` (see `tools/scripts/create_partitions.py`).

## Change log (v1, found while implementing and testing the service)
* `shared.current_guest_id()` / `shared.current_device_id()` + GUCs `app.current_guest_id`, `app.current_device_id`: QR guests and robots/kiosks are now first-class RLS principals.
* `identity.find_login()` (SECURITY DEFINER): login and OTP lookup; RLS correctly hides users from anonymous callers.
* `conversation.find_room_by_code()` and `conversation.room_is_joinable()` (SECURITY DEFINER): join-by-QR works for people who aren't members yet.
* `room_members` policies split per command (SELECT / INSERT join / UPDATE self / DELETE admin). Members always see **their own** rows, which `INSERT … RETURNING` needs.
