import { ButtonMol } from "@/v2/features/shared/components/molecules/ButtonMol";
import { useBackend } from "@/v2/services/backend";
import { useMutation, useQueryClient } from "@tanstack/react-query";

interface Props {
  error: string;
  imageId: number;
  onSuccessSubmit?: () => void;
}

export function ProcessFormError({ error, imageId, onSuccessSubmit }: Props) {
  const { backend } = useBackend();
  const queryClient = useQueryClient();
  const { mutate: retryLatestProcess, isPending } = useMutation({
    mutationFn: () =>
      backend
        .post(`/commands/v1/images/${imageId}/process/retry`)
        .then(() => {}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`/queries/v1/app/cards`] });
      onSuccessSubmit && onSuccessSubmit();
    },
  });

  return (
    <div className="space-y-3">
      <p className="overflow-auto text-xs">{error}</p>
      <ButtonMol
        label="Try Again!"
        isLoading={isPending}
        onClick={() => retryLatestProcess()}
      />
    </div>
  );
}
