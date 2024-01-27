import { useBackend } from "../../../services/backend";
import { paths } from "../../../services/backend/endpoints";
import { BorderBox } from "../../atoms/BorderBox";
import { SvgIcon } from "../../atoms/SvgIcon";
import { useMutation, useQueryClient } from "@tanstack/react-query";

interface Props {
  id: number;
  src: string;
  name: string;
  source: { width: number; height: number };
}

export function ImageCard({ id, src, name, source }: Props) {
  const { backend } = useBackend();
  const queryClient = useQueryClient();
  const { mutate: deleteImage } = useMutation({
    mutationFn: (id: number) =>
      backend
        .delete<
          paths["/images/{id}"]["delete"]["responses"]["200"]["content"]["application/json"]
        >(`/images/${id}`)
        .then((_) => _.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["/images"] }),
  });

  return (
    <BorderBox className="flex h-20 w-72 items-center justify-between p-3 text-xs">
      <div className="flex space-x-3">
        <img src={src} alt={name} className="h-14 w-14 rounded-lg" />
        <div className="flex w-32 flex-col justify-between">
          <label className="truncate">{name}</label>
          <span>
            Source: {source.width}x{source.height}px
          </span>
          <span className="font-bold">Target: -</span>
        </div>
      </div>
      <div className="flex items-center space-x-3">
        <SvgIcon type="run" className="h-6 w-6 cursor-pointer" />
        <SvgIcon
          type="delete"
          className="h-6 w-6 cursor-pointer"
          onClick={() => deleteImage(id)}
        />
      </div>
    </BorderBox>
  );
}
