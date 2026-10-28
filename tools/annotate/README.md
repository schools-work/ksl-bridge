# Annotation tool backend

A thin FastAPI sub-application (`server.py`) mounted at `/annotate-ui` in development. It uses **only** the public use cases of `app.modules.annotation.application`.

* `GET /annotate-ui/next?kind=gloss`: next open task for the annotator (priority from active learning)
* `POST /annotate-ui/{task_id}`: submit glosses, time spans, translation, non-manual tags
* Annotators see **face-blurred** derivatives unless the task kind is `non_manual`.
* Exports ELAN-compatible `.eaf` for offline annotation (`export_eaf`).
