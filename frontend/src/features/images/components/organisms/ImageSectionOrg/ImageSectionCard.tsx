import { ProcessButtonOrg } from "@/features/processes/components/organisms/ProcessButtonOrg";
import { BorderBoxAtm } from "@/features/shared/components/atoms/BorderBoxAtm";
import { SvgIconAtm } from "@/features/shared/components/atoms/SvgIconAtm";
import { useBackend } from "@/services/backend";
import { components } from "@/services/backend/endpoints";
import { useMutation, useQueryClient } from "@tanstack/react-query";

interface Props {
  card: components["schemas"]["CardMod"];
}

export function ImageSectionCard({ card }: Props) {
  const { backend } = useBackend();
  const queryClient = useQueryClient();
  const { mutate: deleteImage } = useMutation({
    mutationFn: () =>
      backend
        .delete(`/commands/v1/images/${card.image_id}/delete`)
        .then(() => {}),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["/queries/v1/app/cards"] }),
  });

  return (
    <BorderBoxAtm className="flex h-20 w-72 items-center justify-between p-3 text-xs">
      <div className="flex space-x-3">
        <img
          src={card.thumbnail_src}
          alt={card.name}
          className="h-14 w-14 rounded-lg"
        />
        <div className="flex w-32 flex-col justify-between">
          <label className="truncate">{card.name}</label>
          <span className="truncate">
            Source: {card.source.width}x{card.source.height}px
          </span>
          <span className="truncate font-bold">
            {card.target
              ? `Target: ${card.target.width}x${card.target.height}px`
              : "Target: -"}
          </span>
        </div>
      </div>
      <div className="flex items-center space-x-3">
        <ProcessButtonOrg card={card} />
        <SvgIconAtm
          type="delete"
          className="h-6 w-6 cursor-pointer"
          onClick={() => deleteImage()}
        />
      </div>
    </BorderBoxAtm>
  );
}
