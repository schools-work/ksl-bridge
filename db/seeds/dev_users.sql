-- DEV ONLY. Never run in staging/production.
-- Usage: psql -v app_password="'change-me'" -f db/seeds/dev_users.sql   (defaults to 'change-me')
\if :{?app_password}
\else
\set app_password '''change-me'''
\endif

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ksl_app_login') THEN
    CREATE ROLE ksl_app_login LOGIN IN ROLE ksl_app;
  END IF;
END $$;
ALTER ROLE ksl_app_login PASSWORD :app_password;

-- password for every dev user below: "KslBridge-dev-2026" (argon2id)
INSERT INTO identity.users (id, email, display_name, preferred_in, preferred_out, dominant_hand, password_hash, email_verified_at)
VALUES
  ('00000000-0000-4000-8000-0000000000a1', 'admin@dev.ksl',     'Dev Admin',        'text', 'text', NULL,    '$argon2id$v=19$m=65536,t=3,p=2$ZQxEd2I1skwJPst82qFkWg$6s3vQOV5XKBjdmcQwliRL7RYNAHPEYT5gFGCUyyyhWc', now()),
  ('00000000-0000-4000-8000-0000000000b1', 'signer@dev.ksl',    'Wanjiku (signer)', 'sign', 'sign', 'right', '$argon2id$v=19$m=65536,t=3,p=2$ZQxEd2I1skwJPst82qFkWg$6s3vQOV5XKBjdmcQwliRL7RYNAHPEYT5gFGCUyyyhWc', now()),
  ('00000000-0000-4000-8000-0000000000b2', 'signer2@dev.ksl',   'Otieno (signer)',  'sign', 'text', 'left',  '$argon2id$v=19$m=65536,t=3,p=2$ZQxEd2I1skwJPst82qFkWg$6s3vQOV5XKBjdmcQwliRL7RYNAHPEYT5gFGCUyyyhWc', now()),
  ('00000000-0000-4000-8000-0000000000c1', 'speaker@dev.ksl',   'Achieng (speaker)','speech','text', NULL,   '$argon2id$v=19$m=65536,t=3,p=2$ZQxEd2I1skwJPst82qFkWg$6s3vQOV5XKBjdmcQwliRL7RYNAHPEYT5gFGCUyyyhWc', now()),
  ('00000000-0000-4000-8000-0000000000d1', 'operator@dev.ksl',  'Demo Operator',    'text', 'text', NULL,    '$argon2id$v=19$m=65536,t=3,p=2$ZQxEd2I1skwJPst82qFkWg$6s3vQOV5XKBjdmcQwliRL7RYNAHPEYT5gFGCUyyyhWc', now()),
  ('00000000-0000-4000-8000-0000000000e1', 'annotator@dev.ksl', 'Deaf Reviewer',    'sign', 'sign', 'right', '$argon2id$v=19$m=65536,t=3,p=2$ZQxEd2I1skwJPst82qFkWg$6s3vQOV5XKBjdmcQwliRL7RYNAHPEYT5gFGCUyyyhWc', now()),
  ('00000000-0000-4000-8000-0000000000f1', 'ml@dev.ksl',        'ML Engineer',      'text', 'text', NULL,    '$argon2id$v=19$m=65536,t=3,p=2$ZQxEd2I1skwJPst82qFkWg$6s3vQOV5XKBjdmcQwliRL7RYNAHPEYT5gFGCUyyyhWc', now())
ON CONFLICT (id) DO NOTHING;

INSERT INTO identity.user_roles (user_id, role) VALUES
  ('00000000-0000-4000-8000-0000000000a1', 'admin'),
  ('00000000-0000-4000-8000-0000000000b1', 'signer'),
  ('00000000-0000-4000-8000-0000000000b2', 'signer'),
  ('00000000-0000-4000-8000-0000000000c1', 'speaker'),
  ('00000000-0000-4000-8000-0000000000d1', 'operator'),
  ('00000000-0000-4000-8000-0000000000e1', 'annotator'),
  ('00000000-0000-4000-8000-0000000000e1', 'deaf_reviewer'),
  ('00000000-0000-4000-8000-0000000000f1', 'ml_engineer')
ON CONFLICT DO NOTHING;

INSERT INTO privacy.consent_texts (scope, version, language, body) VALUES
  ('store_transcripts',    '2026-11-01', 'en', 'I agree that the text of my conversations in this room may be stored for 30 days.'),
  ('contribute_landmarks', '2026-11-01', 'en', 'I agree that my body landmarks (not video) may be used to train KSL Bridge models.'),
  ('contribute_video',     '2026-11-01', 'en', 'I agree that my signing video may be stored and used to train KSL Bridge models.'),
  ('contribute_audio',     '2026-11-01', 'en', 'I agree that my voice recordings may be used to train KSL Bridge speech models.'),
  ('personalisation',      '2026-11-01', 'en', 'I agree that KSL Bridge may learn my personal signing style to understand me better.'),
  ('research_publication', '2026-11-01', 'en', 'I agree that anonymised results including my data may be published in research.'),
  ('demographics',         '2026-11-01', 'en', 'I agree to share demographic information used only to check fairness.')
ON CONFLICT DO NOTHING;
