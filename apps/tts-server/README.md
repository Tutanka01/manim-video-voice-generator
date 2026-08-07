# MOSS TTS Server

Microservice GPU optionnel pour PromptLoom. Il expose
[OpenMOSS-Team/MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5)
par HTTP dans un conteneur Docker. Le checkpoint (~8B, 16-18 Go de VRAM) n'est
**pas chargé en permanence** : il est chargé à la première synthèse (cache miss),
puis déchargé de la VRAM après une grâce d'inactivité
(`TTS_SERVER_MODEL_IDLE_SECONDS`, défaut 15 min) et chargé à nouveau à la
demande. L'application `apps/video-api` l'utilise avec le moteur vocal
`moss-remote`.

Ce comportement évite de consommer ~16-18 Go de VRAM 24/7 quand le serveur dort
entre deux jobs. Le profil de synthèse (device/dtype/attention/fréquence) est
résolu au démarrage sans charger les poids, donc le cache audio CAS fonctionne
même modèle froid : un job entièrement en cache ne relance jamais le GPU.
Mettre `TTS_SERVER_MODEL_IDLE_SECONDS=0` restaure l'ancien comportement
« modèle gardé chaud en VRAM en permanence ».

Le worker vidéo chargeait auparavant le checkpoint MOSS dans chaque processus et
pour chaque job. Déporter la synthèse sur un GPU dédié évite des rechargements
par job, et le cache audio partagé fait qu'une réparation ne resynthétise que
les segments modifiés.

## API

| Méthode | Chemin | Description |
| --- | --- | --- |
| `GET` | `/healthz` | État du moteur (`idle`/`loading`/`ready`/`error`), `model_loaded`, grâce d'inactivité, révisions épinglées, profil de synthèse, GPU/VRAM et profondeur de file. `200` dès que le serveur opérationnel (`idle`/`loading`/`ready`), `503` uniquement en erreur. Sans authentification. |
| `POST` | `/v1/tts/batch` | Soumet tous les segments d'une vidéo. Répond `202` avec un `job_id`. |
| `GET` | `/v1/jobs/{job_id}` | Progression par segment et URLs de téléchargement. |
| `GET` | `/v1/jobs/{job_id}/audio/{key}.wav` | Télécharge le WAV PCM16 canonique d'un segment. |
| `GET` | `/v1/jobs/{job_id}/audio/{key}.mp3` | Compatibilité : encode et publie le MP3 atomiquement à la première demande. |
| `POST` | `/v1/tts` | Synthèse synchrone d'un segment, réservée aux tests ; renvoie un WAV. |

Tous les endpoints `/v1/*` exigent `Authorization: Bearer <key>` ou
`X-API-Key: <key>` lorsque `TTS_SERVER_API_KEYS` est défini.

### Requête batch

```json
{
  "language": "en",
  "model": "OpenMOSS-Team/MOSS-TTS-v1.5",
  "model_revision": "cdd3b911b1585e3f2dbc7775ef10f9926f58850a",
  "consistent_voice": true,
  "reference_audio_b64": null,
  "segments": [
    {"key": "Scene1_IntroEN", "text": "The kernel sits between hardware and programs."},
    {"key": "Scene2_SyscallEN", "text": "A system call crosses that boundary."}
  ]
}
```

- `model` et `model_revision` sont des contrôles optionnels : une valeur
  différente du modèle ou du commit chargé renvoie `409`.
- `consistent_voice` synthétise les segments dans l'ordre. Le WAV fourni, ou le
  premier segment généré, devient la référence vocale des suivants.
- `reference_audio_b64` fournit un WAV de référence en base64 pour tous les
  segments. Le worker vidéo peut y envoyer son premier segment local afin
  qu'une réparation conserve le même timbre.

### État d'un job (extrait)

```json
{
  "job_id": "8c1f…",
  "status": "completed",
  "model_revision": "cdd3b911b1585e3f2dbc7775ef10f9926f58850a",
  "segments": [
    {"key": "Scene1_IntroEN", "status": "done", "cached": false,
     "synthesis_profile_id": "8b1f…",
     "duration_seconds": 6.42, "wav_url": "/v1/jobs/8c1f…/audio/Scene1_IntroEN.wav",
     "mp3_url": "/v1/jobs/8c1f…/audio/Scene1_IntroEN.mp3"}
  ]
}
```

États d'un job : `queued | running | completed | failed`. États d'un segment :
`pending | running | done | failed`. `cached: true` indique que le WAV vient du
cache CAS durci, sans passage sur le GPU.

`mp3_url` reste exposé pour compatibilité, mais la synthèse ne produit aucun MP3.
La première requête vers cette URL encode le WAV canonique dans un fichier
temporaire, puis le publie par renommage atomique. Le pipeline vidéo PromptLoom
ne consomme que les WAV et n'encode qu'une fois, en AAC dans le MP4 final.

Chaque segment expose son `wav_url` dès qu'il atteint `done` : le client
n'attend pas la fin du batch. PromptLoom télécharge progressivement chaque WAV
dans `.wav.part`, valide le PCM16 mono 24 kHz complet, puis le publie par
renommage atomique. Une erreur terminale du serveur fait toujours échouer
explicitement le job vidéo.

### Fast-path du cache MOSS

Lorsque la référence vocale est déjà connue, les hits CAS sont matérialisés dès
l'admission du batch et leurs URLs deviennent immédiatement disponibles. Seuls
les misses entrent dans la FIFO GPU.

Sans référence explicite et avec `consistent_voice=true`, le premier segment
reste une barrière car son WAV détermine les clés suivantes. S'il est lui-même
un hit, il résout immédiatement cette référence et permet de tester le reste du
batch. Un batch entièrement en cache passe directement à `completed` et
n'augmente jamais `queue_depth`.

### Clé de cache durcie

Le `synthesis_profile_id` versionné couvre :

- les commits épinglés du modèle, du code distant et du codec ;
- le digest immuable de l'image et les versions du runtime ;
- le dtype et le backend d'attention résolus ;
- les paramètres de génération, de budget de tokens et de batching ;
- la langue, le texte exact, espaces compris, et le SHA-256 de la référence ;
- le format de sortie PCM.

Toute variation produit donc un cache miss froid au lieu de réutiliser un audio
ambigu. L'endpoint synchrone renvoie cet identifiant dans
`X-Synthesis-Profile-ID` ; `/healthz` expose l'identifiant moteur
`engine_profile_id`.

## Déploiement sur le serveur GPU

Prérequis : pilote NVIDIA, Docker,
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
et au moins 24 Go de VRAM. Le checkpoint BF16 utilise environ 16 à 18 Go.

```bash
cd apps/tts-server
cp .env.example .env        # définir la clé API et le digest immuable
docker compose up --build -d
docker compose logs -f tts  # suivre le téléchargement et le chargement
```

Le premier démarrage résout le profil de synthèse (codec, processeur,
device/dtype/attention) dans le volume `tts_data` (`/data/hf-cache`) ; le
checkpoint lui-même est téléchargé et chargé à la **première synthèse**. Le
serveur est opérationnel dès le profile résolu, avec le modèle `idle` ; le
healthcheck ne limite plus le démarrage au téléchargement du checkpoint.

```bash
curl http://localhost:8100/healthz   # state: idle | loading | ready | error
```

Smoke test d'une synthèse réelle : la première requête après le boot (ou après
une période d'inactivité) attend le chargement du checkpoint, généralement
quelques minutes selon le réseau.

```bash
curl -s -X POST http://localhost:8100/v1/tts \
  -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' \
  -d '{"text":"GPU synthesis is up.","language":"en"}' -o /tmp/test.wav
ffprobe /tmp/test.wav
```

## Connexion de video-api

Dans le `.env` racine de la machine qui exécute le worker :

```bash
VIDEO_API_VOICE_ENGINE=moss-remote
VIDEO_API_TTS_SERVER_URL=http://<gpu-host>:8100
VIDEO_API_TTS_SERVER_API_KEY=<key>
```

Si le serveur est inaccessible ou si le job TTS échoue, le job vidéo **échoue
avec une erreur explicite** dans `logs/voice.log`. Il n'existe aucun fallback
silencieux vers une autre voix.

## Configuration

Voir `.env.example`. Variables principales :

| Variable | Défaut | Rôle |
| --- | --- | --- |
| `TTS_SERVER_API_KEYS` | vide | Clés séparées par des virgules. Vide désactive l'authentification ; réseau de confiance uniquement. |
| `TTS_SERVER_MODEL` | `OpenMOSS-Team/MOSS-TTS-v1.5` | Modèle servi ; chargé à la demande, déchargé après la grâce d'inactivité. |
| `TTS_SERVER_MODEL_REVISION` | `cdd3b9…58850a` | Commit exact sur 40 caractères pour les poids et le code distant. Tags et branches refusés. |
| `TTS_SERVER_CODEC_MODEL` | `OpenMOSS-Team/MOSS-Audio-Tokenizer` | Dépôt du codec chargé par le processeur MOSS. |
| `TTS_SERVER_CODEC_REVISION` | `3cd226…ba782` | Commit exact du codec sur 40 caractères. Tags et branches refusés. |
| `TTS_SERVER_IMAGE_DIGEST` | vide | Identité immuable de l'image déployée : `sha256:<64-hex>` ou `image@sha256:<64-hex>`. Requise pour réutiliser le CAS entre redémarrages. |
| `TTS_SERVER_DEVICE` / `TTS_SERVER_DTYPE` | `auto` / `auto` | Sur le serveur GPU, `auto` sélectionne CUDA et BF16. |
| `TTS_SERVER_MAX_NEW_TOKENS` | `4096` | Plafond absolu. Un plafond par segment est aussi dérivé de la longueur du texte. |
| `TTS_SERVER_BATCH_SIZE` | `1` | Segments de même référence par passe. Augmenter prudemment et valider l'audio. |
| `TTS_SERVER_JOB_TTL_HOURS` | `48` | Délai avant purge des jobs terminaux et de leurs WAV. |
| `TTS_SERVER_CACHE_TTL_DAYS` | `30` | Rétention du cache WAV ; `0` le conserve indéfiniment. |
| `TTS_SERVER_MODEL_IDLE_SECONDS` | `900` | Grâce d'inactivité : après ce délai sans synthèse, le checkpoint est déchargé de la VRAM et rechargé à la demande. `0` garde le modèle chaud en permanence (ancien comportement). |
| `TTS_SERVER_FAKE_ENGINE` | `0` | `1` active le moteur de test, sans GPU ni Torch. |

## Tests

```bash
cd apps/tts-server
docker compose run --rm test
```

Ou localement sans Docker ni GPU, avec le moteur factice :

```bash
uv venv && uv pip install -e '.[test]' && uv run pytest -q
```

## Notes d'exploitation

- Le GPU est sérialisé : une synthèse à la fois, avec file et verrou.
  `queue_depth` ne compte que les jobs avec des misses non résolus. Les jobs
  entièrement en cache contournent la file.
- `VIDEO_API_WORKER_CONCURRENCY=1` correspond à cette sérialisation. Une
  concurrence supérieure ajoute simplement les misses à la file du serveur.
- Un job interrompu par un redémarrage devient
  `failed: interrupted by server restart`. Le retry du worker vidéo réutilise
  ensuite les segments déjà présents dans le CAS.
- L'échantillonnage MOSS est stochastique : le cache stabilise aussi les prises
  vocales entre réparations. Sans `TTS_SERVER_IMAGE_DIGEST` immuable, une
  identité aléatoire propre au démarrage empêche tout hit après redémarrage,
  même si les anciens fichiers restent sur disque.
- `flash-attn` est optionnel. Sans lui, le moteur utilise PyTorch SDPA.
- Le modèle vit en VRAM seulement quand il sert une synthèse. Après
  `TTS_SERVER_MODEL_IDLE_SECONDS` sans génération, il est déchargé (GC +
  `torch.cuda.empty_cache()`) ; le profil de synthèse et les clés CAS
  survivent au déchargement, donc un réemploi ne re-synthétise que les vrais
  misses. Un chargement à la demande (dont le premier après le boot) peut
  prendre quelques minutes ; le worker vidéo (`VIDEO_API_TTS_SERVER_TIMEOUT`,
  défaut 3600 s) tolère ce délai.
