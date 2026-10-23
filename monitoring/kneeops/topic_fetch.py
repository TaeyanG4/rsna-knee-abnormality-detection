"""Isolated, read-only Kaggle SDK calls for discussion topics.

`kaggle competitions topics show` (CLI 2.2.4) omits the opening post and
flattens reply threads. The SDK's GetTopic + ListComments return the opening
post and the full comment tree. Prints one JSON object per topic per line.
"""
import json
import re
import sys
import time

MAX_COMMENT_PAGES = 40


def fetch(svc, tid):
    from kagglesdk.discussions.types.discussions_api_service import ApiGetTopicRequest, ApiListCommentsRequest
    request = ApiGetTopicRequest(); request.id = tid
    topic = (svc.get_topic(request).to_dict() or {}).get('topic') or {}
    comments, token, pages = [], '', 0
    while pages < MAX_COMMENT_PAGES:
        page = ApiListCommentsRequest(); page.topic_id = tid; page.page_size = 100
        if token: page.page_token = token
        data = svc.list_comments(page).to_dict() or {}
        batch = data.get('comments') or []
        comments.extend(batch); pages += 1
        token = data.get('nextPageToken') or data.get('next_page_token') or ''
        if not token or not batch: break
    return {'id': tid, 'topic': topic, 'comments': comments, 'pages': pages, 'complete': not token}


def main():
    if not sys.argv[1:] or not all(x.isdecimal() for x in sys.argv[1:]): raise ValueError('Topic IDs must be numeric')
    from kaggle import api
    with api.build_kaggle_client() as client:
        svc = client.discussions.discussion_api_client
        for tid in map(int, sys.argv[1:]):
            for attempt in range(3):
                try:
                    result = fetch(svc, tid); break
                except Exception as exc:
                    message = str(exc)
                    if attempt < 2 and re.search(r'429|50[234]|timed out|Connection', message):
                        time.sleep(2 * (attempt + 1)); continue
                    if '403' in message: message = 'Kaggle이 이 토론의 접근을 허용하지 않았습니다 (403).'
                    elif '404' in message: message = 'Kaggle에서 이 토론을 찾을 수 없습니다 (404).'
                    result = {'id': tid, 'error': re.sub(r'(KGAT_|kagat_|kagrt_)\S+', '[REDACTED]', message)[:500]}
                    break
            print(json.dumps(result, ensure_ascii=False), flush=True)
            time.sleep(.25)


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print(re.sub(r'(KGAT_|kagat_|kagrt_)\S+', '[REDACTED]', str(exc))[:700], file=sys.stderr)
        sys.exit(1)
