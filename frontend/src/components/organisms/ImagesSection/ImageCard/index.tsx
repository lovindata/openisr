import { useBackend } from "../../../../services/backend";
import { components, paths } from "../../../../services/backend/endpoints";
import { BorderBox } from "../../../atoms/BorderBox";
import { SvgIcon } from "../../../atoms/SvgIcon";
import { ProcessIcon } from "./ProcessIcon";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

interface Props {
  image: components["schemas"]["ImageODto"];
}

export function ImageCard({ image }: Props) {
  const { backend } = useBackend();
  const queryClient = useQueryClient();
  const { mutate: deleteImage } = useMutation({
    mutationFn: () =>
      backend
        .delete<
          paths["/images/{id}"]["delete"]["responses"]["200"]["content"]["application/json"]
        >(`/images/${image.id}`)
        .then((_) => _.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["/images"] }),
  });
  const { data: latestProcess } = useQuery({
    queryKey: [`/images/${image.id}/process`],
    queryFn: () =>
      backend
        .get<
          paths["/images/{id}/process"]["get"]["responses"]["200"]["content"]["application/json"]
        >(`/images/${image.id}/process`)
        .then((_) => _.data),
    refetchInterval: (query) =>
      !query.state.data || query.state.data.status.ended ? false : 1000,
  });

  return (
    <BorderBox className="flex h-20 w-72 items-center justify-between p-3 text-xs">
      <div className="flex space-x-3">
        <img
          src={image.src.thumbnail}
          alt={image.name}
          className="h-14 w-14 rounded-lg"
        />
        <div className="flex w-32 flex-col justify-between">
          <label className="truncate">{image.name}</label>
          <span className="truncate">
            Source: {image.source.width}x{image.source.height}px
          </span>
          <span className="truncate font-bold">
            {latestProcess && !latestProcess.status.ended
              ? `Target: ${latestProcess.target.width}x${latestProcess.target.height}px`
              : "Target: -"}
          </span>
        </div>
      </div>
      <div className="flex items-center space-x-3">
        <ProcessIcon image={image} latestProcess={latestProcess} />
        <SvgIcon
          type="delete"
          className="h-6 w-6 cursor-pointer"
          onClick={() => deleteImage()}
        />
      </div>
    </BorderBox>
  );
}
