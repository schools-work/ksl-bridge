-- =====================================================================================================
--  Executable schema invariants for db/ksl_bridge_v1.sql. Runs in ONE transaction and ROLLS BACK.
--  Usage: psql "$URL" -v ON_ERROR_STOP=1 -f db/checks/schema_invariants.sql
--  Every `expect_error` must be rejected by the database; every `assert_count` must hold.
-- =====================================================================================================
\set ON_ERROR_STOP 1
BEGIN;
-- helpers
CREATE OR REPLACE FUNCTION pg_temp.expect_error(sql text, label text) RETURNS void LANGUAGE plpgsql AS $$
BEGIN
  BEGIN EXECUTE sql; EXCEPTION WHEN OTHERS THEN RAISE NOTICE 'PASS (rejected) %: %', label, SQLERRM; RETURN; END;
  RAISE EXCEPTION 'FAIL: % was accepted', label;
END $$;

INSERT INTO identity.users (id,email,display_name) VALUES
 ('11111111-1111-4111-8111-111111111111','signer@x.ke','Wanjiku'),
 ('22222222-2222-4222-8222-222222222222','owner@x.ke','Owner'),
 ('33333333-3333-4333-8333-333333333333','other@x.ke','Other');
-- optimistic lock
UPDATE identity.users SET display_name='W' WHERE id='11111111-1111-4111-8111-111111111111';
SELECT 'version now', version FROM identity.users WHERE id='11111111-1111-4111-8111-111111111111';
SELECT pg_temp.expect_error($$UPDATE identity.users SET version=99 WHERE email='owner@x.ke'$$,'manual version set');
SELECT pg_temp.expect_error($$INSERT INTO identity.users(display_name) VALUES ('nobody')$$,'user without contact');

-- refresh token rotation + reuse detection
INSERT INTO identity.refresh_tokens(user_id,family_id,token_hash,expires_at) VALUES
 ('11111111-1111-4111-8111-111111111111','aaaaaaaa-0000-4000-8000-000000000000',digest('t1','sha256'),now()+interval '1 day');
SELECT * FROM identity.rotate_refresh_token(digest('t1','sha256'),digest('t2','sha256'));
SELECT * FROM identity.rotate_refresh_token(digest('t1','sha256'),digest('t3','sha256'));  -- reuse
SELECT 'family revoked', count(*) FILTER (WHERE revoked_at IS NOT NULL), count(*) FROM identity.refresh_tokens;

-- venue, room, members, utterance
INSERT INTO venue.venues(id,name,venue_type,owner_id) VALUES ('44444444-4444-4444-8444-444444444444','Java House Demo','restaurant','22222222-2222-4222-8222-222222222222');
INSERT INTO conversation.rooms(id,venue_id,created_by_user,context_profile) VALUES ('55555555-5555-4555-8555-555555555555','44444444-4444-4444-8444-444444444444','11111111-1111-4111-8111-111111111111','restaurant');
SELECT 'join code', join_code FROM conversation.rooms;
INSERT INTO identity.guests(id) VALUES ('66666666-6666-4666-8666-666666666666');
INSERT INTO conversation.room_members(room_id,user_id,role,modality_in,modality_out) VALUES ('55555555-5555-4555-8555-555555555555','11111111-1111-4111-8111-111111111111','signer','sign','sign');
INSERT INTO conversation.room_members(room_id,guest_id,role,modality_in,modality_out) VALUES ('55555555-5555-4555-8555-555555555555','66666666-6666-4666-8666-666666666666','speaker','speech','text');
SELECT pg_temp.expect_error($$INSERT INTO conversation.room_members(room_id,user_id,guest_id,role,modality_in,modality_out) VALUES ('55555555-5555-4555-8555-555555555555','33333333-3333-4333-8333-333333333333','66666666-6666-4666-8666-666666666666','observer','text','text')$$,'member with two identities');
INSERT INTO conversation.utterances(id,created_at,room_id,source,language,glosses,text_en,confidence) VALUES
 ('77777777-7777-4777-8777-777777777777', now(), '55555555-5555-4555-8555-555555555555','sign','ksl','{IX-1,WANT,WATER}','I want water',0.83);
SELECT 'utterance partition', tableoid::regclass FROM conversation.utterances;

-- feedback composite FK + checks
INSERT INTO feedback.events(utterance_id,utterance_created_at,user_id,kind,corrected_glosses)
 SELECT id,created_at,'11111111-1111-4111-8111-111111111111','correct','{IX-1,WANT,FOOD}' FROM conversation.utterances;
SELECT pg_temp.expect_error($$INSERT INTO feedback.events(utterance_id,utterance_created_at,user_id,kind) SELECT id,created_at,'11111111-1111-4111-8111-111111111111','correct' FROM conversation.utterances$$,'correction without content');
SELECT pg_temp.expect_error($$INSERT INTO feedback.events(utterance_id,utterance_created_at,user_id,kind,clip_uri) SELECT id,created_at,'11111111-1111-4111-8111-111111111111','confirm','s3://x' FROM conversation.utterances$$,'clip without consent');

-- consent + dataset invariants
INSERT INTO privacy.consent_texts(scope,version,language,body) VALUES ('contribute_landmarks','2026-11-01','en','...'),('personalisation','2026-11-01','en','...');
INSERT INTO dataset.signers(signer_code,source_id,dominant_hand) SELECT 'S0001',id,'right' FROM dataset.sources WHERE slug='ksl-bridge-own';
INSERT INTO dataset.signers(signer_code,source_id,dominant_hand) SELECT 'S0002',id,'left'  FROM dataset.sources WHERE slug='ksl-bridge-own';
SELECT pg_temp.expect_error($$INSERT INTO dataset.samples(source_id,signer_code,kind,landmarks_uri) SELECT id,'S0001','isolated','s3://a' FROM dataset.sources WHERE slug='ksl-bridge-own'$$,'own sample without consent');
INSERT INTO privacy.consents(id,signer_code,scope,text_version,text_language) VALUES ('88888888-8888-4888-8888-888888888888','S0001','contribute_landmarks','2026-11-01','en');
INSERT INTO dataset.samples(id,source_id,signer_code,consent_id,kind,landmarks_uri,glosses)
 SELECT '99999999-0000-4000-8000-000000000001',id,'S0001','88888888-8888-4888-8888-888888888888','isolated','s3://a','{WATER}' FROM dataset.sources WHERE slug='ksl-bridge-own';
INSERT INTO dataset.samples(id,source_id,signer_code,kind,landmarks_uri,glosses,external_id)
 SELECT '99999999-0000-4000-8000-000000000002',id,'S0001','isolated','s3://b','{FOOD}','ek-1' FROM dataset.sources WHERE slug='ekitabu-ksl-video';
INSERT INTO dataset.samples(id,source_id,signer_code,kind,landmarks_uri,glosses,external_id)
 SELECT '99999999-0000-4000-8000-000000000003',id,'S0002','isolated','s3://c','{FOOD}','ek-2' FROM dataset.sources WHERE slug='ekitabu-ksl-video';
INSERT INTO dataset.builds(id,name,version,purpose,split_seed,code_commit) VALUES ('aaaaaaaa-1111-4111-8111-111111111111','ksl-islr','v1','islr','seed','abc123');
INSERT INTO dataset.build_members(build_id,sample_id,split) VALUES ('aaaaaaaa-1111-4111-8111-111111111111','99999999-0000-4000-8000-000000000001','train'),('aaaaaaaa-1111-4111-8111-111111111111','99999999-0000-4000-8000-000000000003','test');
SELECT pg_temp.expect_error($$INSERT INTO dataset.build_members(build_id,sample_id,split) VALUES ('aaaaaaaa-1111-4111-8111-111111111111','99999999-0000-4000-8000-000000000002','test')$$,'signer leakage across splits');
UPDATE dataset.builds SET frozen_at=now();
SELECT pg_temp.expect_error($$DELETE FROM dataset.build_members WHERE sample_id='99999999-0000-4000-8000-000000000001'$$,'mutate frozen build');
SELECT * FROM dataset.v_build_composition;

-- ml registry, gates, promotion, exclusion constraint
INSERT INTO ml.experiments(id,task,name,hypothesis,research_question) VALUES ('bbbbbbbb-1111-4111-8111-111111111111','cslr','cslr-baseline','MPM pretraining lowers WER','RQ2');
INSERT INTO ml.training_runs(id,experiment_id,task,dataset_build_id,config,config_sha256,code_commit,seed,status) VALUES
 ('cccccccc-1111-4111-8111-111111111111','bbbbbbbb-1111-4111-8111-111111111111','cslr','aaaaaaaa-1111-4111-8111-111111111111','{}',digest('{}','sha256'),'abc123',42,'succeeded');
INSERT INTO ml.model_versions(id,task,name,version,run_id,format,artifact_uri,sha256) VALUES
 ('dddddddd-0000-4000-8000-000000000001','cslr','cslr_conformer_ctc','2026.11.0','cccccccc-1111-4111-8111-111111111111','pytorch','s3://m1',digest('m1','sha256')),
 ('dddddddd-0000-4000-8000-000000000002','cslr','cslr_conformer_ctc','2026.11.1','cccccccc-1111-4111-8111-111111111111','pytorch','s3://m2',digest('m2','sha256'));
SELECT pg_temp.expect_error($$INSERT INTO ml.model_versions(task,name,version,format,artifact_uri,sha256) VALUES ('asr','x','local-1','pytorch','s3://x','\x00')$$,'non-bootstrap model without run');
INSERT INTO ml.model_versions(task,name,version,is_bootstrap,bootstrap_source,format,artifact_uri,sha256) VALUES ('asr','asr_bootstrap','local-1',true,'open-weights ASR (local)','pytorch','s3://asr','\x00');
SELECT pg_temp.expect_error($$SELECT ml.promote_model('dddddddd-0000-4000-8000-000000000001','server','22222222-2222-4222-8222-222222222222')$$,'promote without gates');
INSERT INTO ml.release_gates(model_version_id,gate,passed) SELECT m, g, true FROM unnest(ARRAY['dddddddd-0000-4000-8000-000000000001','dddddddd-0000-4000-8000-000000000002']::uuid[]) m, unnest(ARRAY['quality','no_regression','fairness','latency','safety_set','licence']) g;
SELECT 'promoted 1', ml.promote_model('dddddddd-0000-4000-8000-000000000001','server','22222222-2222-4222-8222-222222222222') IS NOT NULL;
SELECT 'promoted 2', ml.promote_model('dddddddd-0000-4000-8000-000000000002','server','22222222-2222-4222-8222-222222222222') IS NOT NULL;
SELECT 'live', name, version, stage FROM ml.v_live_models;
SELECT pg_temp.expect_error($$INSERT INTO ml.deployments(model_version_id,task,stage,traffic_fraction) VALUES ('dddddddd-0000-4000-8000-000000000001','cslr','production',1)$$,'two production models overlapping');
INSERT INTO ml.evaluations(id,model_version_id,dataset_build_id,split,suite,code_commit) VALUES ('eeeeeeee-1111-4111-8111-111111111111','dddddddd-0000-4000-8000-000000000002','aaaaaaaa-1111-4111-8111-111111111111','test','fairness','abc');
INSERT INTO ml.evaluation_results(evaluation_id,metric,slice_key,slice_value,value,n) VALUES
 ('eeeeeeee-1111-4111-8111-111111111111','wer','all','all',0.24,500),('eeeeeeee-1111-4111-8111-111111111111','wer','dominant_hand','right',0.22,400),('eeeeeeee-1111-4111-8111-111111111111','wer','dominant_hand','left',0.31,100);
SELECT metric, slice_key, round(gap::numeric,3), worst_slice_if_error_metric FROM ml.v_fairness_gaps;
SELECT pg_temp.expect_error($$INSERT INTO ml.decoder_configs(task,beam,bias_max_nats) VALUES ('cslr',8,5)$$,'strong biasing');

-- inference + clarification
INSERT INTO inference.requests(task,model_version_id,deployment_stage,status,utterance_id) VALUES ('cslr','dddddddd-0000-4000-8000-000000000002','production','ok','77777777-7777-4777-8777-777777777777');
INSERT INTO inference.clarifications(utterance_id,utterance_created_at,kind,options,p_best,expected_loss,venue_type) SELECT id,created_at,'choose_one','{WATER,WAIT}',0.62,0.38,'restaurant' FROM conversation.utterances;
SELECT * FROM feedback.v_quality_by_model;

-- personalisation consent
SELECT pg_temp.expect_error($$INSERT INTO feedback.personal_adaptations(user_id,kind,base_model_id,embedding,n_examples) VALUES ('11111111-1111-4111-8111-111111111111','prototype','dddddddd-0000-4000-8000-000000000002',array_fill(0.1,ARRAY[256])::vector,3)$$,'adaptation without consent');
INSERT INTO privacy.consents(user_id,scope,text_version,text_language) VALUES ('11111111-1111-4111-8111-111111111111','personalisation','2026-11-01','en');
INSERT INTO feedback.personal_adaptations(user_id,kind,base_model_id,embedding,n_examples) VALUES ('11111111-1111-4111-8111-111111111111','prototype','dddddddd-0000-4000-8000-000000000002',array_fill(0.1,ARRAY[256])::vector,3);

-- agent: sessions, tool guard
INSERT INTO agent.definitions(id,name,version,venue_type,intents,slot_prompts) VALUES ('ffffffff-1111-4111-8111-111111111111','restaurant-ordering',1,'restaurant','{"order_food":["item","qty"]}','{}');
INSERT INTO agent.sessions(id,room_id,definition_id,venue_id) VALUES ('ffffffff-2222-4222-8222-222222222222','55555555-5555-4555-8555-555555555555','ffffffff-1111-4111-8111-111111111111','44444444-4444-4444-8444-444444444444');
SELECT pg_temp.expect_error($$INSERT INTO agent.sessions(room_id,definition_id) VALUES ('55555555-5555-4555-8555-555555555555','ffffffff-1111-4111-8111-111111111111')$$,'second active session in room');
INSERT INTO agent.tool_calls(session_id,tool_id,idempotency_key,arguments,status) SELECT 'ffffffff-2222-4222-8222-222222222222',id,'k1','{"name":"water"}','succeeded' FROM agent.tools WHERE name='menu.lookup';
SELECT pg_temp.expect_error($$INSERT INTO agent.tool_calls(session_id,tool_id,idempotency_key,arguments) SELECT 'ffffffff-2222-4222-8222-222222222222',id,'k2','{}' FROM agent.tools WHERE name='order.add'$$,'write tool not bound to venue');
INSERT INTO agent.venue_tools(venue_id,tool_id,adapter) SELECT '44444444-4444-4444-8444-444444444444',id,'demo_restaurant' FROM agent.tools WHERE name='order.add';
INSERT INTO agent.tool_calls(session_id,tool_id,idempotency_key,arguments) SELECT 'ffffffff-2222-4222-8222-222222222222',id,'k2','{"item_id":21,"qty":1}' FROM agent.tools WHERE name='order.add';
SELECT pg_temp.expect_error($$UPDATE agent.tool_calls SET status='succeeded' WHERE idempotency_key='k2'$$,'unconfirmed write tool success');
UPDATE agent.tool_calls SET status='succeeded', confirmed_by_user=true WHERE idempotency_key='k2';
SELECT pg_temp.expect_error($$INSERT INTO agent.tools(name,description,input_schema,output_schema,side_effect,requires_confirmation) VALUES ('robot.dance','x','{}','{}','physical',false)$$,'physical tool without confirmation');

-- knowledge: hybrid search + TM
INSERT INTO knowledge.documents(id,venue_id,kind,title,content_sha256) VALUES ('12121212-1212-4212-8212-121212121212','44444444-4444-4444-8444-444444444444','menu','Menu',digest('m','sha256'));
INSERT INTO knowledge.chunks(document_id,section_path,ordinal,content,embedding) VALUES
 ('12121212-1212-4212-8212-121212121212','Menu > Drinks',0,'Menu item: water. Price: KSh 100',array_fill(0.1,ARRAY[384])::vector),
 ('12121212-1212-4212-8212-121212121212','Menu > Food',1,'Menu item: chicken burger. Price: KSh 650',array_fill(0.2,ARRAY[384])::vector);
SELECT 'chunk venue filled', count(*) FROM knowledge.chunks WHERE venue_id IS NOT NULL;
SELECT section_path, round(score::numeric,3) FROM knowledge.hybrid_search('44444444-4444-4444-8444-444444444444','chicken burger price',array_fill(0.2,ARRAY[384])::vector,2);
INSERT INTO knowledge.translation_memory(glosses,text_en,domain,source) VALUES ('{IX-1,WANT,WATER}','I want water','restaurant','annotation');
SELECT gloss_key FROM knowledge.translation_memory;

-- jobs claim
INSERT INTO ops.jobs(kind) VALUES ('train');
SELECT 'claimed', (ops.claim_job('train','w1')).status;

-- audit append-only
SELECT pg_temp.expect_error($$DELETE FROM audit.events$$,'delete audit');

-- erasure
SELECT privacy.erase_user('11111111-1111-4111-8111-111111111111','22222222-2222-4222-8222-222222222222');
SELECT 'after erase', status, email, (SELECT count(*) FROM privacy.consents WHERE user_id='11111111-1111-4111-8111-111111111111' AND withdrawn_at IS NULL) FROM identity.users WHERE id='11111111-1111-4111-8111-111111111111';
SELECT 'licence risk', slug, usage FROM dataset.v_licence_risk ORDER BY slug;

-- ---------------------------------------------------------------- RLS is fail-closed (as the app role)
CREATE OR REPLACE FUNCTION pg_temp.assert_count(sql text, expected int, label text) RETURNS void LANGUAGE plpgsql AS $$
DECLARE n int;
BEGIN
  EXECUTE sql INTO n;
  IF n <> expected THEN RAISE EXCEPTION 'FAIL %: got %, expected %', label, n, expected; END IF;
  RAISE NOTICE 'PASS %: %', label, n;
END $$;
INSERT INTO identity.users (id, email, display_name) VALUES ('abababab-1111-4111-8111-111111111111', 'member@x.ke', 'Member');
INSERT INTO conversation.room_members (room_id, user_id, role, modality_in, modality_out)
  VALUES ('55555555-5555-4555-8555-555555555555', 'abababab-1111-4111-8111-111111111111', 'signer', 'sign', 'sign');
SET LOCAL ROLE ksl_app;
SELECT set_config('app.current_user_id', '', true), set_config('app.current_venue_id', '', true);
SELECT pg_temp.assert_count('SELECT count(*) FROM conversation.rooms', 0, 'no GUC → no rooms');
SELECT pg_temp.assert_count('SELECT count(*) FROM knowledge.chunks', 0, 'no GUC → no venue chunks');
SELECT set_config('app.current_user_id', 'abababab-1111-4111-8111-111111111111', true);
SELECT pg_temp.assert_count('SELECT count(*) FROM conversation.rooms', 1, 'member sees room');
SELECT pg_temp.assert_count('SELECT count(*) FROM identity.users', 1, 'user sees only self');
SELECT set_config('app.current_user_id', '33333333-3333-4333-8333-333333333333', true);
SELECT pg_temp.assert_count('SELECT count(*) FROM conversation.rooms', 0, 'outsider sees nothing');
SELECT set_config('app.current_user_id', '', true), set_config('app.current_venue_id', '44444444-4444-4444-8444-444444444444', true);
SELECT pg_temp.assert_count('SELECT count(*) FROM knowledge.chunks', 2, 'venue sees its chunks');
SELECT pg_temp.expect_error($$DELETE FROM audit.events$$, 'app role cannot delete audit');
RESET ROLE;

ROLLBACK;
