from fastapi import APIRouter, HTTPException

from app.schemas import MusicLinksRequest, MusicLinksResponse
from app.services.music_chain import recommend_top3_youtube_links

router = APIRouter(prefix="/ai", tags=["music"])


@router.post("/song", response_model=MusicLinksResponse)
async def create_music_links(payload: MusicLinksRequest) -> MusicLinksResponse:
    try:
        links = await recommend_top3_youtube_links(
            [song.model_dump() for song in payload.recent_songs]
        )
        if len(links) != 3:
            raise HTTPException(status_code=502, detail="유튜브 링크를 3개 생성하지 못했습니다.")
        return MusicLinksResponse(youtube_links=links)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="음악 추천 처리 중 오류가 발생했습니다.") from exc