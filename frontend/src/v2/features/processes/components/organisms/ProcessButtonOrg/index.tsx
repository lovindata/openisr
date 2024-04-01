import { ProcessButtonIcon } from "@/v2/features/processes/components/organisms/ProcessButtonOrg/ProcessButtonIcon";
import { ProcessFormOrg } from "@/v2/features/processes/components/organisms/ProcessFormOrg";
import { useModal } from "@/v2/features/shared/hooks/contexts/Modal/useModal";
import { useBackend } from "@/v2/services/backend";
import { components } from "@/v2/services/backend/endpoints";
import { useMutation, useQueryClient } from "@tanstack/react-query";

interface Props {
  card: components["schemas"]["CardMod"];
}

export function ProcessButtonOrg({ card }: Props) {
  const { backend } = useBackend();
  const queryClient = useQueryClient();
  const { mutate: stopProcess } = useMutation({
    mutationFn: () =>
      backend
        .delete(`/command/v1/images/${card.image_id}/process/stop`)
        .then(() => {}),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["/query/v1/app/cards"] }),
  });

  const { openModal, closeModal } = useModal();

  switch (card.status.type) {
    case "Runnable":
      return (
        <ProcessButtonIcon
          type="run"
          onClick={() =>
            openModal(
              <ProcessFormOrg card={card} onSuccessSubmit={closeModal} />
            )
          }
        />
      );
    case "Stoppable":
      return (
        <ProcessButtonIcon
          type="stop"
          duration={card.status.duration}
          onClick={() => stopProcess()}
        />
      );
    case "Errored":
      return (
        <ProcessButtonIcon
          type="error"
          duration={card.status.duration}
          onClick={() =>
            openModal(
              <ProcessFormOrg card={card} onSuccessSubmit={closeModal} />
            )
          }
        />
      );
    case "Downloadable":
      return (
        <a href={card.status.image_src} download>
          <ProcessButtonIcon type="download" />
        </a>
      );
  }
}
